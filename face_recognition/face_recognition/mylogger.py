import logging


log_path=r'E:\ros2_ws\src\logs.log'
# 创建一个日志记录器
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 创建一个日志格式器
formatter = logging.Formatter('[%(levelname)s] [%(asctime)s]: %(message)s')

# 创建一个终端日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 创建一个文件日志处理器
file_handler = logging.FileHandler(log_path, mode='a',encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)
mylogger=logger