import argparse
from deepface import DeepFace
from mtcnn import MTCNN
from glob import glob
import cv2
import os
import tsne

def create_folder(dir, name):
	folder = ""
	processed = False
	if not os.path.exists(os.path.join(dir, name)):
		folder = os.path.join(dir, name)
	else:
		processed = True
		i = 1
		while os.path.exists(os.path.join(dir, name + str(i))):
			i += 1
		folder = os.path.join(dir, name + str(i))
	
	return folder, processed
	

def process_video(v_path, o_path='output/', t_interval=5.0, min_face_size=240, min_face_w=64, min_face_h=64, face_detect_threshold=0.5, worksheet="", skip_processed=False):
	detector = MTCNN(min_face_size=min_face_size)

	count = 0
	vidcap = cv2.VideoCapture(v_path)
	success,image = vidcap.read()
	success = True
	total_secs = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / vidcap.get(cv2.CAP_PROP_FPS)
	video_name = os.path.basename(v_path).split(".")[0]
	target, processed = create_folder(o_path, video_name)

	if skip_processed and processed:
		print(f"{v_path} has been processed in {o_path}, skipping this video.")
		return
	else:
		os.mkdir(target)

	has_worksheet = False
	frames = []
	if worksheet != "":
		video_name = os.path.basename(v_path).split(".")[0]
		with open(worksheet, "r") as f:
			lines = f.readlines()
			f.close()
		for line in lines:
			if has_worksheet == False:
				has_worksheet = True
				continue
			asp = line.split(",")
			l_name = asp[0]
			frame_number = int(asp[1])
			if l_name == video_name and frame_number not in frames:
				frames.append(frame_number)
			
		print(f"Found {len(frames)} predefined frames in the worksheet to process")

	while success:
		pos = count * t_interval # seconds
		if pos > total_secs:
			break
		vidcap.set(cv2.CAP_PROP_POS_MSEC, pos * 1000)
		success,image = vidcap.read()
		if not success:
			count = count + 1
			continue
		if has_worksheet and count not in frames:
			count = count + 1
			continue

		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		face_objs = detector.detect_faces(image)
		new_lines = ""

		for i, face in enumerate(face_objs):
			confidence = face["confidence"]
			if confidence < face_detect_threshold:
				continue
			x, y, w, h = face["box"]
			if w < min_face_w or h < min_face_h:
				continue

			face = image[y:y+h, x:x+w]
			embedding = DeepFace.represent(img_path=face, model_name="Facenet512", enforce_detection=False)[0]['embedding']
			cv2.imwrite(f"{target}/t{count}f{i}.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

			new_lines += f"{video_name}\t{count}\t{i}\t{[x, y, w, h]}\t{embedding}\n"
		
		with open(f"{target}/embeddings.tsv", "a") as f:
			f.write(new_lines)
			f.close()

		count = count + 1
		print(count)

	vidcap.release()

	tsne.csv_dimensionality_reducing(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--input', type=str, default='', help='path to an mp4 video or a folder with mp4 videos')
    parser.add_argument('--output', type=str, default='output/', help='path to output')
    parser.add_argument('--time_interval', type=float, default=5.0, help='path to output')
    parser.add_argument('--min_face_size', type=int, default=200, help='the minimum face size for detection (width * height)')
    parser.add_argument('--min_face_w', type=int, default=0, help='the minimum face width for detection')
    parser.add_argument('--min_face_h', type=int, default=0, help='the minimum face height for detection')
    parser.add_argument('--face_detect_threshold', type=float, default=0.5, help='the confidence threshold for face detection')
    parser.add_argument('--worksheet', type=str, default='', help='a csv file with two columns: video_name, frame_number to indicate which frames to process (the frame number is senstive to the time interval set, and the first line should be kept for headers)')
    parser.add_argument('--skip_processed', action='store_true', help='Skip videos that can be found in the same output path')

    opt = parser.parse_args()
    if os.path.isfile(opt.input): process_video(opt.input, opt.output, opt.time_interval, min_face_size=opt.min_face_size, min_face_w=opt.min_face_w, min_face_h=opt.min_face_h, face_detect_threshold=opt.face_detect_threshold, worksheet=opt.worksheet, skip_processed=opt.skip_processed)
    elif os.path.isdir(opt.input):
	    for video_name in glob(opt.input + "/*.mp4"):
		    process_video(video_name, opt.output, opt.time_interval, min_face_size=opt.min_face_size, min_face_w=opt.min_face_w, min_face_h=opt.min_face_h, face_detect_threshold=opt.face_detect_threshold, worksheet=opt.worksheet, skip_processed=opt.skip_processed)