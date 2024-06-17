from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd


def init_ff(dataset='all',phase='test'):#根据指定的数据集和阶段初始化 FaceForensics++ 数据集的文件路径。#
	# #phase: 指定阶段（train、val、test），用于获取文件路径
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']#用来判断dataset的值是否在这些数据中
	original_path='data/FaceForensics++/original_sequences/youtube/c23/videos/'#修改代码，将raw改成c23
	#原始路径data/FaceForensics++/original_sequences/youtube/c23/videos/
	#original_path: FaceForensics++ 数据集中 'youtube' 类别原始视频的路径
	folder_list = sorted(glob(original_path+'*'))#获得指定文件夹下所有路径，original_path + '*'，表示匹配指定路径下的所有文件和文件夹。

	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]#包含原始视频路径的列表 image_list，并为每个路径设置了标签 0，表示这是原始视频。
	label_list=[0]*len(image_list)


	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/c23/videos/'#修改代码，将raw改成c23
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list



	return image_list,label_list



def init_dfd():
	real_path='data/FaceForensics++/original_sequences/actors/raw/videos/*.mp4'
	real_videos=sorted(glob(real_path))
	fake_path='data/FaceForensics++/manipulated_sequences/DeepFakeDetection/raw/videos/*.mp4'
	fake_videos=sorted(glob(fake_path))

	label_list=[0]*len(real_videos)+[1]*len(fake_videos)

	image_list=real_videos+fake_videos

	return image_list,label_list


def init_dfdc():
		
	label=pd.read_csv('data/DFDC/labels.csv',delimiter=',')
	folder_list=[f'data/DFDC/videos/{i}' for i in label['filename'].tolist()]
	label_list=label['label'].tolist()
	
	return folder_list,label_list


def init_dfdcp(phase='test'):

	phase_integrated={'train':'train','val':'train','test':'test'}

	all_img_list=[]
	all_label_list=[]

	with open('data/DFDCP/dataset.json') as f:
		df=json.load(f)
	fol_lab_list_all=[[f"data/DFDCP/{k.split('/')[0]}/videos/{k.split('/')[-1]}",df[k]['label']=='fake'] for k in df if df[k]['set']==phase_integrated[phase]]
	name2lab={os.path.basename(fol_lab_list_all[i][0]):fol_lab_list_all[i][1] for i in range(len(fol_lab_list_all))}
	fol_list_all=[f[0] for f in fol_lab_list_all]
	fol_list_all=[os.path.basename(p)for p in fol_list_all]
	folder_list=glob('data/DFDCP/method_*/videos/*/*/*.mp4')+glob('data/DFDCP/original_videos/videos/*/*.mp4')
	folder_list=[p for p in folder_list if os.path.basename(p) in fol_list_all]
	label_list=[name2lab[os.path.basename(p)] for p in folder_list]
	

	return folder_list,label_list




def init_ffiw():
	# assert dataset in ['real','fake']
	path='data/FFIW/FFIW10K-v1-release/'
	folder_list=sorted(glob(path+'source/val/videos/*.mp4'))+sorted(glob(path+'target/val/videos/*.mp4'))
	label_list=[0]*250+[1]*250
	return folder_list,label_list



def init_cdf():

	image_list=[]
	label_list=[]

	video_list_txt='data/Celeb-DF-v2/List_of_testing_videos.txt'
	with open(video_list_txt) as f:
		
		folder_list=[]#定义了一个空列表用来存储视频路径
		for data in f:#循环遍历文件的每一行，对每一行进行分割和处理，然后将得到的视频文件夹的路径添加到 folder_list 中。
			# print(data)
			line=data.split()#从文件中读取的一行数据（data）按照空格进行分割，并将分割后的元素存储在列表 line 中。
			#print(line)
			path=line[1].split('/')#将 line 列表中的第二个元素（索引为 1）按照斜杠 / 进行分割，并将分割后的结果存储在列表 path 中。
			#folder_list+=['data/Celeb-DF-v2/'+path[0]+'/videos/'+path[1]]
			folder_list += ['data/Celeb-DF-v2/' + path[0] + '/' + path[1]]
			label_list+=[1-int(line[0])]
		return folder_list,label_list
		


