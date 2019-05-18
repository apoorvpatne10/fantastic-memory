from lipreading.videos import Video
from lipreading.visualization import show_video_subtitle
from lipreading.helpers import labels_to_text
from core.decoders import Decoder
from utils.spell import Spell
from model import LipNet
from keras.optimizers import Adam
from keras import backend as K
from os import listdir
from os.path import isfile, join
from scipy import misc
import numpy as np
import sys
import os
import cv2
import dlib
import skvideo.io
import argparse

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = 'common/predictors/shape_predictor_68_face_landmarks.dat'

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = 'common/dictionaries/grid.txt'

def predict(video_path, weight_path, absolute_max_string_len=32, output_size=28):
    print("\nLoading data from disk...")
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print("Data loaded.\n")

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape


    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path = PREDICT_DICTIONARY)
    decoder = Decoder(greedy = PREDICT_GREEDY, beam_width = PREDICT_BEAM_WIDTH,
                      postprocessors = [labels_to_text, spell.sentence])

    X_data = np.array([video.data]).astype(np.float32)/255
    input_length = np.array([len(video.data)])

    y_pred = lipnet.predict(X_data)
    result = decoder.decode(y_pred, input_length)[0]

    return (video, result)


parser = argparse.ArgumentParser(description='Add path to video')
parser.add_argument('path', help='Mention the path to the video')
args = parser.parse_args()

FPS = 25
FRAME_ROWS = 120
FRAME_COLS = 120
NFRAMES = 5 # size of input volume of frames
MARGIN = NFRAMES/2
COLORS = 1 # grayscale
CHANNELS = COLORS*NFRAMES
MAX_FRAMES_COUNT= 250 # corresponding to 10 seconds, 25Hz*10

mouth_destination_path = os.path.dirname('demo_data'+'/' + 'mouth')
if not os.path.exists(mouth_destination_path):
    os.makedirs(mouth_destination_path)

def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    assert FPS == video_fps, "video FPS is not 25 Hz"

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = skvideo.io.FFmpegWriter('demo_data/'+input_video_path.split('/')[-1]+'_lip_highlight.mp4')

    predictor_path = 'common/predictors/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    activation = []
    max_counter = MAX_FRAMES_COUNT
    num_frames = min(total_num_frames,max_counter)
    counter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Required parameters for mouth extraction.
    width_crop_max = 0
    height_crop_max = 0

    temp_frames = np.zeros((num_frames,FRAME_ROWS,FRAME_COLS),dtype="uint8")
    for i in np.arange(num_frames):
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if ret==0:
            break
        if counter > num_frames:
            break

                # Detection of the frame
        detections = detector(frame, 1)

        # 20 mark for mouth
        marks = np.zeros((2, 20))

        # All unnormalized face features.
        if len(detections) > 0:
            for k, d in enumerate(detections):

                # Shape of the face.
                shape = predictor(frame, d)

                co = 0
                # Specific for the mouth.
                for ii in range(48, 68):
                    """
                    This for loop is going over all mouth-related features.
                    X and Y coordinates are extracted and stored separately.
                    """
                    X = shape.part(ii)
                    A = (X.x, X.y)
                    marks[0, co] = X.x
                    marks[1, co] = X.y
                    co += 1

                # Get the extreme points(top-left & bottom-right)
                X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                    int(np.amax(marks, axis=1)[0]),
                                                    int(np.amax(marks, axis=1)[1])]

                # Find the center of the mouth.
                X_center = (X_left + X_right) / 2.0
                Y_center = (Y_left + Y_right) / 2.0

                # Make a boarder for cropping.
                border = 30
                X_left_new = X_left - border
                Y_left_new = Y_left - border
                X_right_new = X_right + border
                Y_right_new = Y_right + border

                # Width and height for cropping(before and after considering the border).
                width_new = X_right_new - X_left_new
                height_new = Y_right_new - Y_left_new
                width_current = X_right - X_left
                height_current = Y_right - Y_left

                # Determine the cropping rectangle dimensions(the main purpose is to have a fixed area).
                if width_crop_max == 0 and height_crop_max == 0:
                    width_crop_max = width_new
                    height_crop_max = height_new
                else:
                    width_crop_max += 1.5 * np.maximum(width_current - width_crop_max, 0)
                    height_crop_max += 1.5 * np.maximum(height_current - height_crop_max, 0)

                # # # Uncomment if the lip area is desired to be rectangular # # # #
                #########################################################
                # Find the cropping points(top-left and bottom-right).
                X_left_crop = int(X_center - width_crop_max / 2.0)
                X_right_crop = int(X_center + width_crop_max / 2.0)
                Y_left_crop = int(Y_center - height_crop_max / 2.0)
                Y_right_crop = int(Y_center + height_crop_max / 2.0)
                #########################################################

                if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < w and Y_right_crop < h:
                    mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]

                    # Save the mouth area.
                    mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                    mouth_gray = cv2.resize(mouth_gray,(FRAME_COLS,FRAME_ROWS))
                    temp_frames[i*COLORS:i*COLORS+COLORS,:,:] = mouth_gray
                    cv2.imwrite(mouth_destination_path + '/' + 'frame' + '_' + str(counter) + '.png', mouth_gray)

                    print("The cropped mouth is detected ...")
                    activation.append(1)
                else:
                    cv2.putText(frame, 'The full mouth is not detectable. ', (30, 30), font, 1, (0, 255, 255), 2)
                    temp_frames[i*COLORS:i*COLORS+COLORS,:,:] = np.zeros((FRAME_ROWS,FRAME_COLS),dtype="uint8")
                    print("The full mouth is not detectable....")
                    activation.append(0)
        else:
            cv2.putText(frame, 'Mouth is not detectable. ', (30, 30), font, 1, (0, 0, 255), 2)
            print("Mouth is not detectable. ...")
            temp_frames[i*COLORS:i*COLORS+COLORS,:,:] = np.zeros((FRAME_ROWS,FRAME_COLS),dtype="uint8")
            activation.append(0)

        if activation[counter] == 1:
            # Demonstration of face.
            cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)

        # cv2.imshow('frame', frame)
        print(f'frame number {counter} of {num_frames}')

        # write the output frame to file
        print(f"writing frame {counter + 1} with activation {activation[counter]}")
        writer.writeFrame(frame)
        counter += 1
    writer.close()
    temp_frames = temp_frames.astype(np.uint8)
    print('temp_frames shape: ', temp_frames.shape)
    skvideo.io.vwrite('demo_data/'+input_video_path.split('/')[-1]+'_lip_cropped.mp4', temp_frames)
    vidctr = 0
    SAMPLE_LEN = int(num_frames-2*MARGIN)
    visual_cube = np.zeros((SAMPLE_LEN,CHANNELS,FRAME_ROWS,FRAME_COLS), dtype="uint8")
    for i in np.arange(MARGIN,SAMPLE_LEN+MARGIN):
    		visual_cube[vidctr,:,:,:] = temp_frames[int(COLORS*(i-MARGIN)):int(COLORS*(i+MARGIN)),:,:]
    		vidctr = vidctr+1
    visual_cube = visual_cube.transpose(0, 2, 3, 1)
    print(f'visual_cube shape: {visual_cube.shape}   , sequence_length = {visual_cube.shape[0]} ')

def create_numpy_sequence():
    MAX_WIDTH = 90
    MAX_HEIGHT = 90

    path = 'demo_data'
    filelist = sorted(os.listdir(path + '/'))
    sequence = []
    for i, img_name in enumerate(filelist):
        if i % 2 == 0:
            if img_name.startswith('frame') and img_name.endswith('.png'):
                image = misc.imread(path + '/' + img_name)
                image = misc.imresize(image, (MAX_WIDTH, MAX_HEIGHT))
                sequence.append(image)
    sequence = np.stack(sequence, axis=0)
    data_dir = 'frame_numpy_sequence'
    np.save(data_dir + '/video', sequence)
    return sequence


if __name__ == '__main__':

    if len(sys.argv) == 2:
        video_name = sys.argv[1]
        process_video(video_name)
        sequence = create_numpy_sequence()
        videox, result = predict(sys.argv[1], 'models/overlapped-weights368.h5')
    else:
        sys.exit()

    with open('predicted_text.txt', 'w') as fr:
        fr.write(result + '\n' + str(video_name) + "_lip_highlight.mp4")

    stripe = "-" * len(result)
    print(f"                  --{stripe}- ")
    print(f"[ Phrase :   ]   |> {result} |")
    print(f"                  --{stripe}- ")
