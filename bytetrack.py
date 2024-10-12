import numpy as np
from kalman_filter import KalmanFilter
from hungarian import linear_assignment


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

def get_tlwh(stracks):
    return [strack.tlwh() for strack in stracks]

def iou_distance(atlbrs, btlbrs):
    # 将输入从 (x_center, y_center, w, h) 转换为 (x_min, y_min, x_max, y_max)
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float64)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float64)
    N, M = atlbrs.shape[0], btlbrs.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float64)  
    # (x_min, y_min, x_max, y_max) = (x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2)
    atlbrs[:, 0] = atlbrs[:, 0] - atlbrs[:, 2] / 2
    atlbrs[:, 1] = atlbrs[:, 1] - atlbrs[:, 3] / 2
    atlbrs[:, 2] = atlbrs[:, 0] + atlbrs[:, 2]
    atlbrs[:, 3] = atlbrs[:, 1] + atlbrs[:, 3]
    
    btlbrs[:, 0] = btlbrs[:, 0] - btlbrs[:, 2] / 2
    btlbrs[:, 1] = btlbrs[:, 1] - btlbrs[:, 3] / 2
    btlbrs[:, 2] = btlbrs[:, 0] + btlbrs[:, 2]
    btlbrs[:, 3] = btlbrs[:, 1] + btlbrs[:, 3]

    overlaps = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        inter_x1 = np.maximum(atlbrs[i, 0], btlbrs[:, 0])
        inter_y1 = np.maximum(atlbrs[i, 1], btlbrs[:, 1])
        inter_x2 = np.minimum(atlbrs[i, 2], btlbrs[:, 2])
        inter_y2 = np.minimum(atlbrs[i, 3], btlbrs[:, 3])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        a_area = (atlbrs[i, 2] - atlbrs[i, 0]) * (atlbrs[i, 3] - atlbrs[i, 1])
        b_area = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
        union_area = a_area + b_area - inter_area
        overlaps[i, :] = np.where(union_area > 0, inter_area / union_area, 0.0)
    return 1 - overlaps

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(get_tlwh(stracksa), get_tlwh(stracksb))
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


class STrack:
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):        
        self.state = TrackState.New

        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, track_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score

    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def tlwh_to_xyah(self, tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track._tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    
class BYTETracker(object):
    def __init__(self, frame_rate=30, low_conf_thresh=0.1, high_conf_thresh=0.5):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.det_low_conf_thresh = low_conf_thresh
        self.det_high_conf_thresh = high_conf_thresh

        self.frame_id = 0
        self.track_id = 0
        self.max_time_lost = frame_rate

    def update(self, det_results):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = []
        inds_low = []
        for i, det in enumerate(det_results):
            if det.score > self.det_high_conf_thresh:
                remain_inds.append(i)
            elif det.score > self.det_low_conf_thresh:
                inds_low.append(i)
        detections = []
        for i in remain_inds:
            detections.append(STrack([det_results[i].left, 
                                      det_results[i].top, 
                                      det_results[i].width, 
                                      det_results[i].height, 
                                      ], det_results[i].score))
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = iou_distance(get_tlwh(strack_pool), get_tlwh(detections))
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)
        print("A"*40, matches)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)
        
        detections_second = []
        for i in inds_low:
            detections_second.append(STrack([det_results[i].left, 
                                             det_results[i].top, 
                                             det_results[i].width, 
                                             det_results[i].height, 
                                             ], det_results[i].score))
            
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(get_tlwh(r_tracked_stracks), get_tlwh(detections_second))
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        print("B"*40, matches)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = iou_distance(get_tlwh(unconfirmed), get_tlwh(detections))
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        print("C"*40, matches)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_high_conf_thresh:
                continue
            track.activate(KalmanFilter(), self.frame_id, self.track_id)
            self.track_id += 1
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

