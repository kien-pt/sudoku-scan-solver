import cv2
import numpy as np


def getSudoku(thresh, contour_grille):
    points = np.vstack(contour_grille).squeeze()
    sorted_points = sorted(points, key=lambda x: x[0])
    points = np.vstack(sorted_points).squeeze()

    top_left = points[0]
    bottom_left = points[1]
    if points[0][1] > points[1][1]:
        top_left = points[1]
        bottom_left = points[0]

    top_right = points[2]
    bottom_right = points[3]
    if points[2][1] > points[3][1]:
        top_right = points[3]
        bottom_right = points[2]

    pts1 = np.float32([top_left, bottom_left, top_right, bottom_right])
    pts2 = np.float32([[0, 0], [0, 324], [324, 0], [324, 324]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst_thresh = cv2.warpPerspective(thresh, M, (324, 324))
    return dst_thresh


def img_proc(frame, contour_grille, sol):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    points = np.vstack(contour_grille).squeeze()
    sorted_points = sorted(points, key=lambda x: x[0])
    points = np.vstack(sorted_points).squeeze()

    top_left = points[0]
    bottom_left = points[1]
    if points[0][1] > points[1][1]:
        top_left = points[1]
        bottom_left = points[0]

    top_right = points[2]
    bottom_right = points[3]
    if points[2][1] > points[3][1]:
        top_right = points[3]
        bottom_right = points[2]

    pts1 = np.float32([top_left, bottom_left, top_right, bottom_right])
    pts2 = np.float32([[0, 0], [0, 324], [324, 0], [324, 324]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(frame, M, (324, 324))
    dst_thresh = cv2.warpPerspective(thresh, M, (324, 324))

    for i in range(1, 10):
        if not (i == 9):
            width = 2
            cv2.line(dst, (0, i * 36), (324, i * 36), (0, 255, 0), width)
            cv2.line(dst, (i * 36, 0), (i * 36, 324), (0, 255, 0), width)


            #
            for row in range(9):
                for col in range(9):
                    if sol[row][col] > 0:
                        cv2.putText(dst, str(sol[row][col]), (col * 36 + 10, (row + 1) * 36 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # # for row in range(9):
            # #     for col in range(9):
            # #         color = (0, 0, 255)
            # #         if matrix[row][col] == table[row][col]:
            # #             continue
            #         # if table[row][col] > 0:
            #         #     cv2.putText(dst, str(table[row][col]), (col * 35 + 10, (row + 1) * 35 - 10),
            #         #                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            #
            # # cc = cv2.bitwise_not(dst_thresh[35:70, 35:70])
            # # prediction = classify_image(cc)
            # # print(prediction)
            #
    M = cv2.getPerspectiveTransform(pts2, pts1)
    h, w, c = frame.shape
    back = cv2.warpPerspective(dst, M, (w, h))
    img2gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask = mask.astype('uint8')
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    img2_fg = cv2.bitwise_and(back, back, mask=mask).astype('uint8')
    frame = cv2.add(img1_bg, img2_fg)
    cv2.drawContours(frame, [contour_grille], 0, (0, 255, 0), 6)

    return frame