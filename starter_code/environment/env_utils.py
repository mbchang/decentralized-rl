# import cv2

# def render(env, scale):
#     frame = env.render(mode='rgb_array')

#     if frame is not None:
#         h, w, c = frame.shape
#         frame = cv2.resize(frame, dsize=(int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
#         return frame