import cv2
import os
import numpy as np

def process_images(logo_path, knife_image_path):
    logo_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    knife_cover_img = cv2.imread(knife_image_path, cv2.IMREAD_GRAYSCALE)

    knife_cover_img = cv2.rotate(knife_cover_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    
    kp1, des1 = sift.detectAndCompute(logo_img, None)
    kp2, des2 = sift.detectAndCompute(knife_cover_img, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    distances = [m.distance for m, n in matches if m.distance < 0.8 * n.distance]
    if distances:
        threshold = sum(distances) / len(distances) * 0.75
        good_matches = [m for m, n in matches if m.distance < threshold * n.distance]
    else:
        good_matches = []

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        good_matches = [good_matches[i] for i in range(len(matchesMask)) if matchesMask[i]]

    match_img = cv2.drawMatches(logo_img, kp1, knife_cover_img, kp2, good_matches, None, flags=2)

    result_text = "Images match!" if len(good_matches) > 25 else "Images do not match."
    match_text = f"Number of features matched: {len(good_matches)}"
    print(result_text)
    print(match_text)

    cv2.putText(match_img, result_text, (1100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(match_img, match_text, (1100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

    display_scale = 0.4
    display_img = cv2.resize(match_img, None, fx=display_scale, fy=display_scale)
    cv2.imshow('Matched Images', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    base_path =  r"" #Give path for test images (I used a folder which contains subfolders with both the image sets. So i had around 10 folders with 2 images each.)
    test_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

    # Process each folder of images
    for folder in test_folders:
        print(f"Processing folder: {folder}")
        folder_path = os.path.join(base_path, folder)
        files = os.listdir(folder_path)

        if len(files) >= 2:
            logo_path = os.path.join(folder_path, files[1])
            knife_image_path = os.path.join(folder_path, files[0])
            process_images(logo_path, knife_image_path)

        input("Press Enter to continue to the next set of images...")

if __name__ == "__main__":
    main()
