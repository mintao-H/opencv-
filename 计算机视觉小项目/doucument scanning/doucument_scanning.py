#文档扫描
#1.边缘检测
#2.轮廓检测
#3.透视变化
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
class DocumentScanner:
    def __init__(self):
        pass

    def preprocess(self,img):        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #双边滤波降噪
        blur = cv2.bilateralFilter(gray, 3, 200, 200)
       
        # 边缘检测（Canny）
        edged = cv2.Canny(gray, 25, 200)

        #膨胀连接边缘
        edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        #展示图片
        cv2.imshow("edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return edged

    #查找文档轮廓
    def find_document_contour(self,edged):
        #查找轮廓
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #按轮廓面积降序排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #文档轮廓
        vailed_contours = None
        
        for contour in contours:
            #1.面积过滤（至少占图像的5%）
            # area = cv2.contourArea(contour)
            # if area < 0.05 * edged.size:
            #     continue

            #2.周长过滤（避免细长轮廓）
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            #3.必须是凸边形
            # if not cv2.isContourConvex(approx):
            #     continue

            #4.四边形顶点筛选（允许3-6个点，后续优化）
            # if len(approx) < 3 or len(approx) > 6:
            #     continue
            if len(approx) == 4:
                vailed_contours = approx
                break
        return vailed_contours
    
    #对四边形顶点排序
    def order_points(self,pts):
        rect = np.zeros((4, 2),dtype="float32")

        #左上角是x+y最小的点，右下角是x+y最大的点
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        #右下角是x-y最小的点，左上角是x-y最大的点
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def optimize_quad(self,approx):
        #将3-6边形转化为四边形
        hull = cv2.convexHull(approx.reshape(-1, 2))
        # 检查hull中的点数是否少于4
        if len(hull) < 4:
            # 如果点数少于4，直接返回这些点
            return self.order_points(hull.reshape(-1, 2))
        #使用k-means算法聚类合并近邻点
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        _, _, centers = cv2.kmeans(hull.astype(np.float32), 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        #重新排序顶点
        return self.order_points(centers)
    
    #透视变化
    def prespective_transform(self,image,pts):
        rect = self.order_points(pts.reshape(4,2))
        (tl, tr, br, bl) = rect

        #计算新图像的宽度和高度
        width_a = np.linalg.norm(tl - tr)
        width_b = np.linalg.norm(bl - br)
        width_max = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tl - bl)
        height_b = np.linalg.norm(tr - br)
        height_max = max(int(height_a), int(height_b))

        #目标点坐标
        dst = np.array([
            [0, 0],
            [width_max - 1, 0],
            [width_max - 1, height_max - 1],
            [0, height_max - 1]],dtype="float32"
        )

        #计算透视变化矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width_max, height_max))
        return warped

    #后处理（二值化）
    def post_process(self,warped):
        return cv2.adaptiveThreshold(
            cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY),
            255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,5
            )

    #主函数
    def document_scanner(self,image_path):
        #读取图片
        image = cv2.imread(image_path)
        original_image = image.copy()

        if image is None:
            print("Could not read image")
            return
        
        #预处理
        edged = self.preprocess(image)

        #查找轮廓
        screen_contour = self.find_document_contour(edged)
        if screen_contour is None:
            print("Could not find document contour")
            return 
        
        #顶点优化
        #optimized_contour = self.optimize_quad(screen_contour)

        # 绘制轮廓
        cv2.polylines(image, [screen_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Contours", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #透视变化
        warped = self.prespective_transform(original_image, screen_contour)

        #后处理
        final_image = self.post_process(warped)

        #显示结果
        cv2.imshow("Original_image",cv2.resize(original_image,(640,480)))
        cv2.imshow("Final_image",cv2.resize(final_image,(640,480)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return final_image

if __name__ == '__main__':
    scan = DocumentScanner()
    result_img = scan.document_scanner('img\\123.jpeg')
    file_name = "{}.jpg".format(os.getpid())
    cv2.imwrite(file_name,result_img)
    text = pytesseract.image_to_string(Image.open(file_name),lang='chi_sim')
    print(text)
    os.remove(file_name)

    
