import base64
import logging
import os
import sqlite3
import threading
import time
import torch
import chardet
import websockets
import rclpy
from Live_Detection import detect as LD
from models.facenet import Facenet
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from aip import AipFace
import json
import numpy as np
import urllib3
import asyncio
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from concurrent.futures import ThreadPoolExecutor
# ros2 run face_recognition FaceRecognitionPublisher
from .mylogger import logger
CONSTANTS={
    'video_device':0
}
current_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_path)
with open('./Baidu_keys.json','r') as fp:
    keys=json.load(fp)
with open('./room_info.json','r') as fp:
    room_info=json.load(fp) 
imageType = "BASE64"
groupIdList=room_info['room_id']
options={
    'liveness_control':'NORMAL',
    'match_threshold':80
}   
device_id='D058984606732'
# logging.basicConfig(filename=r'./logs.txt',level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
PROTOTXT_PATH=r'./models/deploy_prototxt.txt'
# PROTOTXT_PATH=os.path.join(current_directory,'../models/deploy_prototxt.txt')
MODEL_PATH=r'./models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
# MODEL_PATH=os.path.join(current_directory,'../models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
Live_detect_path=r'./models/80x80_MiniFASNetV2.pth'
Face_net_path=r'./models/facenet_mobilenet.pth'
db_path=r'./face_interfaces.db'
model = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

def network():
    
    try:
        http = urllib3.PoolManager()
        http.request('GET', 'https://baidu.com')
        return True
    except:
        return False

def dnn_face_detect(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    model.setInput(blob)
    faces=np.squeeze(model.forward())

    return faces

def draw_face(frame,faces):
    h,w=frame.shape[:2]
    
    for i in range(0, faces.shape[0]):
        confidence = faces[i, 2]
        if confidence>0.5:
            box = faces[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(int)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code
def detect_has_face(frame):
    faces=dnn_face_detect(frame=frame)
    flag=0
    for i in range(0, faces.shape[0]):
        confidence = faces[i, 2]
        if confidence>0.5:
            flag=1
    return flag
def saveStranger(obj,frame,t):
    file_path=f'./Strangers/{t.tm_year}/{t.tm_mon}/{t.tm_mday}/{t.tm_hour}-{t.tm_min}-{t.tm_sec}.jpg'
    directory = os.path.dirname(file_path)
    # 自动创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(file_path,frame)
    logger.warning('Strangers logged!')

def opendoor():
    pass

class Publisher(Node):
    def __init__(self,name):
        super().__init__(name)
        logger.info("%s Lunched!" % name)
        # self.netType=network()
        self.netType=0
        self.publisher=self.create_publisher(String,'faceAuthentication',10)
        # self.video_publisher=self.create_publisher(Image,'video_stream',10)
        self.video_publisher=self.create_publisher(String,'video_stream',10)
        self.timer=self.create_timer(3,self.time_callback)
        self.cap=cv2.VideoCapture(CONSTANTS['video_device'])
        self.client=AipFace(keys['APP_ID'],keys['API_KEY'],keys['SECRET_KEY'])
        self.cap_thread = threading.Thread(target=self.process_video)
        self.cap_thread.start()
        self.bridge=CvBridge()
        self.faces=None
        self.pause=False
        # self.websocket_uri = f'ws://99.suyiiyii.top:45685/ws/device/{device_id}'
        self.pause_timer=None
    def process_video(self):
        logger.info('Video process start!')
        while rclpy.ok():
            ret,frame=self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            self.faces=dnn_face_detect(frame)
            self.hasFace=draw_face(frame, self.faces)
            # 将 OpenCV 图像转换为 ROS Image 消息并发布
            # image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            small_frame=cv2.resize(frame, dsize=None, fx=0.4, fy=0.4)
            _, buffer = cv2.imencode('.jpg', small_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            msg = String()
            msg.data = jpg_as_text
            self.video_publisher.publish(msg)
            # logger.info('Message sended!')
            cv2.imshow('video capture', frame)
            key = cv2.waitKey(10) & 0xff
            if key == 27:
                self.cap.release()
                cv2.destroyAllWindows()
                break
            time.sleep(1.00/30)
        
    def time_callback(self):
        if self.pause:
            return  # Skip callback execution during the pause
        ret, frame = self.cap.read()
        
        msg=String()
        if self.netType:
            name,statu=self.req_baidu(self.client,frame)
        else:            
            name,statu=self.req_local(frame)
        if statu==1:
            msg.data='known'
            self.publisher.publish(msg)
            # opendoor()
            logger.info(f'Send Message:{msg.data} Name:{name}')
             # Start a 5-second pause using a one-shot timer
            self.pause = True
            self.pause_timer=self.create_timer(10, self.end_pause)  # Ends pause after 10 seconds
        else:
            msg.data='unknown'
            t=time.localtime()
            if statu==-1:
                saveStranger(self,frame,t)
            self.publisher.publish(msg)
            logger.info(f'Send Message:{msg.data} Name:{name}')
        
            
    def end_pause(self):
        self.pause = False  # Resume after the pause is over
        if self.pause_timer is not None:
            self.pause_timer.cancel()
            self.pause_timer=None
    async def start_websocket(self):
        async with websockets.connect(self.websocket_uri) as websocket:
            # await websocket.send(json.dumps({'status': self.device_status}))
            logger.info('WebSocket connected!')
            async for message in websocket:
                # 处理来自服务器的控制命令
                command = json.loads(message).get('command')
                if command:
                    self.handle_command(command)
    
    def handle_command(command):
        print(command)
        pass
        
    def destroy_node(self):
        # 关闭摄像头
        self.cap.release()
        super().destroy_node()
    def req_baidu(self,client:AipFace,frame,):
        
        img_base64=image_to_base64(frame)
        response=client.multiSearch(img_base64,imageType,groupIdList,options)
        # print(response)
        if response["error_msg"]=='SUCCESS':
            result=response['result']
            face=result['face_list'][0]
            logger.info(f"Name:{face['user_list'][0]['user_info']} Score:{face['user_list'][0]['score']:.2f}")
            name=face['user_list'][0]['user_info']
            return name,1
        else:
            # print('Name:Unknown')
            logger.warning(f'Request API failed! Erro:{response["error_msg"]}')
            if(response["error_msg"]=='pic not has face'):
                return None,0
            if(response["error_msg"]=='match user is not found'):
                return None,-1
            return None,-2
    def req_local(self,frame):
        hasFace,face=Get_Face(frame)
        if not hasFace:
            return None,-3
        name=compare_face(face,face_feature_dict)
        if not name:
            return None,-1
        Live_label=LD.Live_Detect(cv2.resize(frame, (80, 80)),Live_detect_path)
        if Live_label!=1:
            logger.warning(f"Name:{name} Non-living body!")
            return name,-2
        
        return name,1
def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    return img

def Get_Face(frame):
    ori_img=frame.copy()
    h,w=frame.shape[:2]
    faces=dnn_face_detect(frame)
    max_len=0
    max_index=-1
    Face_pos=[]
    for i in range(faces.shape[0]):
        confidence=faces[i,2]
        if confidence>0.5:
            start_x, start_y, end_x, end_y=faces[i,3:7]
            len=max(end_x-start_x,end_y-start_y)
            if len>max_len:
                max_index=i
    box = faces[max_index, 3:7] * np.array([w, h, w, h])
    start_x, start_y, end_x, end_y = box.astype(int)
    Face_pos=[start_x,start_y,end_x-start_x+1,end_y-start_y+1]
    
    if max_index>-1:
        return True,get_ROI(*(Face_pos),ori_img)
    else:
        return False,None
    

def get_ROI(x,y,w,h,frame):
    block_len=max(w,h)
    # ROI=np.zeros((block_len,block_len,3),np.uint8)
    x1=(x+w) if ((x+w)<frame.shape[1]) else (frame.shape[1])
    y1=(y+h) if ((y+h)<frame.shape[0]) else (frame.shape[0])
    ROI=frame[y:y1-1,x:x1-1]
    ROI=np.pad(ROI,((0,block_len-(y1-y)),(0,block_len-(x1-x)),(0,0)),'constant')
    # print(f'ROI_point:{x,y,x1-1,y1-1}  Rec+point{x,y,x+w,y+h}')
    # cv2.imshow('ROI', ROI)
    return ROI
# 加载存储的人脸特征
def load_face_feature():
    con=sqlite3.connect(db_path)
    cur=con.cursor()
    cur.execute("select * from faces")
    res={}
    for item in cur:
        name=item[1]
        img_base64=item[4]
        img=base64_to_image(img_base64)
        img_feature=inference(img)
        res[name]=img_feature
    return res
        
# 人脸特征对比
def cosin_metric(x1, x2):
    # single feature vs. single feature
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
# 对比人脸特征返回比对名称，不存在返回none
def compare_face(face_img0, face_feature_dict):
    face_img0_feature = inference(face_img0)
    # print(face_img0_feature)
    max_prob = 0
    max_name = ''
    for name in face_feature_dict.keys():
        face_img1_feature = face_feature_dict[name]
        prob = cosin_metric(face_img0_feature, face_img1_feature)
        if prob > max_prob:
            max_prob = prob
            max_name = name
    # print(max_name, max_prob)

    if max_prob > 0.3:
        return max_name
    else:
        return None
# 获取人脸特征
@torch.no_grad()
def inference(img):
    # img 是人脸的剪切图像
    img = cv2.resize(img, (160, 160))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    feat = net(img).numpy().flatten()
    # print(feat)
    return feat
face_net=torch.load(Face_net_path,map_location=torch.device('cpu'),weights_only=True)
net=Facenet('mobilenet',mode='predict').eval()
net.load_state_dict(face_net,strict=False)
net.eval()
face_feature_dict = load_face_feature()
def main(args=None):
    rclpy.init(args=args)
    node=Publisher('FaceRecognition')
    # node.run()
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(rclpy.spin(node))
    
    # rclpy.spin(node)
    # rclpy.shutdown()
    try:
        while rclpy.ok():
            # 使用 spin_once 来处理回调
            rclpy.spin_once(node, timeout_sec=0.1)  # 超时设置为 0.1 秒
            # 继续执行其他任务
            # time.sleep(0.1)  # 可以根据需要调整此休眠时间
    except KeyboardInterrupt:
        pass
    finally:
        # 清理并关闭节点
        node.destroy_node()
        rclpy.shutdown()
    # asyncio.run(mainTask(args=args))