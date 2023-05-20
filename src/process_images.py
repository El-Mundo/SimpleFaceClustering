import argparse
from deepface import DeepFace
from mtcnn import MTCNN
from glob import glob
import cv2
import os
import tsne
import process_video

SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'bmp']

def process_images(v_path, o_path='output/', min_face_size=240, min_face_w=64, min_face_h=64, face_detect_threshold=0.5, skip_processed=False):
	detector = MTCNN(min_face_size=min_face_size)

	count = 0
	if not v_path.endswith('/'):
		video_name = v_path.split('/')[-1]
	else:
		video_name = v_path.split('/')[-2]
	target, processed = process_video.create_folder(o_path, video_name)

	if skip_processed and processed:
		print(f"{v_path} has been processed in {o_path}, skipping this video.")
		return
	else:
		os.mkdir(target)

	for format in SUPPORTED_FORMATS:
		for image_name in glob(opt.input + "/*.{}".format(format)):
			try:
				image = cv2.imread(image_name)
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
			except Exception as e:
				print(e)
				continue

			count = count + 1
			print(count)

	tsne.csv_dimensionality_reducing(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--input', type=str, default='', help='path to a folder with images')
    parser.add_argument('--output', type=str, default='output/', help='path to output')
    parser.add_argument('--min_face_size', type=int, default=200, help='the minimum face size for detection (width * height)')
    parser.add_argument('--min_face_w', type=int, default=0, help='the minimum face width for detection')
    parser.add_argument('--min_face_h', type=int, default=0, help='the minimum face height for detection')
    parser.add_argument('--face_detect_threshold', type=float, default=0.5, help='the confidence threshold for face detection')
    parser.add_argument('--skip_processed', action='store_true', help='Skip an image folder that can be found in the same output path')

    opt = parser.parse_args()
    if os.path.isdir(opt.input):
	    process_images(opt.input, opt.output, min_face_size=opt.min_face_size, min_face_w=opt.min_face_w, min_face_h=opt.min_face_h, face_detect_threshold=opt.face_detect_threshold, skip_processed=opt.skip_processed)