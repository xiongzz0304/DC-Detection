import json
import cv2

pre_results='/data/xy_data/new_mm/mmdetection-master/results_neg.bbox.json'

json_file='/data/xy_data/datasets/3_21/coco/annotations/instances_test2017.json'

img_path='/data/xy_data/datasets/3_21/coco/test2017/'

save_path='/data/xy_data/datasets/3_29/negs/'

with open(pre_results,'r') as f:
	predicts=json.load(f)

with open(json_file,'r') as f:
	ann=json.load(f)
	img_list=ann['images']

img_names=[]
for img in img_list:
	img_names.append(img['file_name'])
count=0
num=0
for pre in predicts:
	#print(pre['score']-0.75)
	#break
	if (pre['score'] -0.85)>0 :
		#break
		img_id=int(pre['image_id'])
		img_name=img_names[img_id]
	#print(type(img_name))
	#print(img_name)
		image_path=img_path+img_name
		img=cv2.imread(image_path)
		count+=1
		bbox=pre['bbox']
		x=int(bbox[0])
		y=int(bbox[1])
		w=int(bbox[2])
		h=int(bbox[3])
		max_wh=max(w,h)
		xmax=x+max_wh-1
		ymax=y+max_wh-1
		cropped=img[y:ymax,x:xmax]
		imgonly=img_name.split('.')[0]
		cv2.imwrite(save_path+imgonly+str(count)+'.jpg',cropped)
	else:
		print(pre['score'])


print(count)
