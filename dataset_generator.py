import numpy as np
import cv2 as cv
import os

dim = 32
bg_color = np.array([204,204,204])

clear = np.ones((dim,dim,3)) * bg_color
start = -4
end = dim-start 
step = 1

output_folder = "datasets/new_one_ball/"

with open(output_folder+"centroids.csv","w") as centr:

    cnt = 0
    # red loop
    for ri in range(start, end, step):
        for rj in range(start, end,step):
            red = clear.copy()
            cv.circle(red, (rj, ri), 3, (0,0,255), thickness=-1)

            red_seen = 1 if (ri>=-2 and rj>=-2 and ri<=dim+2 and rj<=dim+2) else -1

            # blue loop
            # for bi in range(start, end, step):
            #     for bj in range(start, end, step):
            #         img = red.copy()
            #         cv.circle(img, (bj, bi), 3, (255,0,0), thickness=-1)
            #         img = cv.GaussianBlur(img, (5, 5), 0)

            #         blue_seen = 1 if (bi>=-2 and bj>=-2 and bi<=dim+2 and bj<=dim+2) else -1
            img = red.copy()
            img = cv.GaussianBlur(img, (5, 5), 0)
            encoding = str(rj)+ "," + str(ri)+ "," + str(red_seen)+ "\n"#"\t" + str(bj)+ "\t" + str(bi)+ "\t" + str(blue_seen)+ "\n"
            print(encoding,end="")
            centr.write(encoding)

            number = (5-len(str(cnt)))*"0" + str(cnt)
            filename = f"img_{number}.jpg"

            # Save the image as PNG
            cv.imwrite(os.path.join(output_folder, filename), img)
            cnt+=1
            cv.imshow("img",img)
            cv.waitKey(1)


cv.destroyAllWindows()