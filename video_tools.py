import cv2

__author__ = 'Dandi Chen'

def get_video_len(in_video):
    cap = cv2.VideoCapture(in_video)
    # return int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def vidoe2img(in_video, step = 1):
    cap = cv2.VideoCapture(in_video)
    video_len = get_video_len(in_video)

    out_frames = []
    out_frames_ID = []
    for frame_ID in range(0, video_len, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ID)
        print 'img_idx = ', int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        _, cur_frame = cap.read()
        rgb = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)

        out_frames.append(rgb)
        out_frames_ID.append([frame_ID])

    return out_frames, out_frames_ID