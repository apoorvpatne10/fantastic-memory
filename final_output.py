from lipreading.videos import Video
from lipreading.visualization import show_video_subtitle

FACE_PREDICTOR_PATH = 'common/predictors/shape_predictor_68_face_landmarks.dat'

def main():
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)

    with open('predicted_text.txt', 'r') as fr:
        pred = fr.read()

    pred_text, video_name = pred.split('\n')[0], pred.split('\n')[1]

    video.from_video('demo_data/' + video_name)

    show_video_subtitle(video.face, pred_text)

if __name__ == '__main__':
    main()
