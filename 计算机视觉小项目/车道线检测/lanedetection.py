#1.图像加载
#2.图像预处理：图片灰度化，高斯滤波
#3.canny边缘检测
#4.感兴趣区域检测
#5.霍夫直线检测
#6.直线拟合
#7.车道线叠加
import cv2
import numpy as np
class LaneDetection:
    def __init__(self):
        self.img = None
        self.img_path = None
        self.gray = None
        self.gaussian = None
        self.mask = None
        self.canny = None
        self.lines = None
        self.left_line = None
        self.right_line = None
        self.left_fit = None
        self.right_fit = None
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None
        self.height = None

    def load_image(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(self.img_path)
        self.height, self.width = self.img.shape[:2]
    
    def gray_image(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    
    def gaussian_blur(self):
        self.gaussian = cv2.GaussianBlur(self.gray,(5,5),3)
    
    def canny_edge(self):
        self.canny = cv2.Canny(self.gaussian,50,150)
    
    def mask_image(self):
        """
        对边缘检测图像应用多边形遮罩，仅保留感兴趣区域
        
        处理流程:
        1. 创建与边缘检测图像同尺寸的黑色遮罩层
        2. 定义三角形多边形区域（底边全宽，顶点在图像中心）
        3. 在遮罩层上填充多边形区域为白色
        4. 通过位运算将边缘检测结果限制在遮罩区域内
        5. 可视化并显示处理结果
        
        参数说明:
        self.canny: 输入的边缘检测图像(二值图)
        self.mask: 输出参数，存储处理后的遮罩图像
        
        返回值:
        无，结果保存在self.mask属性中
        """
        # 获取原始图像尺寸
        height, width = self.canny.shape
        
        # 创建初始遮罩层
        mask = np.zeros_like(self.canny)
        
        # 定义三角形顶点坐标（左下，右下，中心点）
        polygon = np.array([[(0, height),(width, height),(width // 2, height // 2)]])
        
        # 在遮罩层填充多边形区域
        cv2.fillPoly(mask, polygon, 255)
        
        # 应用位运算进行区域遮罩
        self.mask = cv2.bitwise_and(self.canny, mask)
        
        # 可视化显示处理结果
        cv2.imshow("mask", self.mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def hough_lines(self):
        """
        使用概率霍夫变换检测图像中的直线段并进行可视化
        
        处理流程:
        1. 调用OpenCV的HoughLinesP函数检测二值掩膜图中的直线段
        2. 遍历检测到的直线段，在原图上用蓝色线条标注
        3. 展示处理结果图像并保持窗口直到按键
        
        参数说明:
        self.mask: 输入的二进制掩膜图像
        rho=1: 距离分辨率(像素)
        theta=np.pi/180: 角度分辨率(弧度)
        threshold=100: 累加器阈值参数
        minLineLength=200: 线段最小长度(像素)
        maxLineGap=400: 线段间最大允许间隔(像素)
        
        Returns:
        None (结果直接存储在self.img和self.lines属性中)
        """
        # 概率霍夫变换检测线段，结果存储在self.lines
        self.lines = cv2.HoughLinesP(self.mask, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=50)
        
        # # 在原始图像上绘制检测到的蓝色线段
        # for line in self.lines:
        #     x1, y1, x2, y2 = line[0]
        #     #cv2.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # 可视化结果并保持显示窗口
        # cv2.imshow("hough_lines", self.img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    def fit_lines(self):
        left_fit, right_fit = [], []
        for idx,line in enumerate(self.lines):
            x1, y1, x2, y2 = line[0]
            print(f"\n处理第{idx+1}条线段：[({x1},{y1})->({x2},{y2})]")

            #阶段1：坐标有效性验证
            #验证坐标范围有效性
            if not all(0 <= val <= self.width if i%2 == 0 else 0 <= val <= self.height
                       for i, val in enumerate([x1, y1, x2, y2])):
                print("坐标超出有效范围，跳过该线段")
                continue

            #计算几何特征
            dx = x2 - x1
            dy = y2 - y1
            distance = np.hypot(dx, dy)

            #验证线段长度
            if distance < 50:
                print("线段过短，跳过该线段")
                continue

            #阶段2：数学可行性检查
            #仅排查垂直线段（x坐标差过小）
            if abs(dx) < 1e-5:
                print("线段垂直，跳过该线段")
                continue

            #阶段3：直线拟合尝试
            try:
                #使用加权最小二乘法提高稳定性
                weights = np.array([distance, distance]) #给长线段更高权重
                parameters = np.polyfit([x1, x2],[y1, y2], 1, w=weights)

                #参数有效性验证
                if np.any(np.isnan(parameters)) or np.any(np.isinf(parameters)):
                    print("参数无效，跳过该线段")
                    continue
                slope, intercept = parameters[0], parameters[1]
                print(f"成功拟合：斜率={slope:.4f}，截距={intercept:.4f}")

            except np.linalg.LinAlgError:
                print("矩阵奇异，无法求解")
                continue
            except Exception as e:
                print(f"未知错误：{e}")
                continue

            #阶段4：车道线分类
            #根据斜率方向分类
            if slope < -0.1:  #左侧车道线阈值
                print("归类为左车道线")
                left_fit.append([slope, intercept])
            elif slope > 0.1:  #右侧车道线阈值
                print("归类为右车道线")
                right_fit.append([slope, intercept])
            else:
                print("水平线段，已忽略")
        
        #阶段5：结果综合处理
        #左车道线处理
        if len(left_fit) > 0:
            self.left_fit = np.median(left_fit, axis=0) #使用中位数更抗干扰
            y_top, y_bottom = self.height//2, self.height
            self.left_x = (np.array([y_top, y_bottom]) - self.left_fit[1]) / self.left_fit[0]
            print(f"左车道线方程：y = {self.left_fit[0]:.2f}x + {self.left_fit[1]:.2f}")
        
        #右车道线处理
        if len(right_fit) > 0:
            self.right_fit = np.median(right_fit, axis=0)
            y_top, y_bottom = self.height//2, self.height
            self.right_x = (np.array([y_top, y_bottom]) - self.right_fit[1]) / self.right_fit[0]
            print(f"右车道线方程：y = {self.right_fit[0]:.2f}x + {self.right_fit[1]:.2f}")
        
        #可视化处理
        output_img = self.img.copy()
        if hasattr(self, "left_x") and self.left_x is not None:
            cv2.line(output_img,
                     (int(self.left_x[0]),self.height//2),
                     (int(self.left_x[1]), self.height),(0,255,0),5)
        if hasattr(self, "right_x") and self.right_x is not None:
            cv2.line(output_img,
                     (int(self.right_x[0]),self.height//2),
                     (int(self.right_x[1]), self.height),(0,255,0),5)
        
        cv2.imshow("fit_lines", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    lane = LaneDetection()
    lane.load_image("assets/2.jpg")
    lane.gray_image()
    lane.gaussian_blur()
    lane.canny_edge()
    lane.mask_image()
    lane.hough_lines()
    lane.fit_lines()