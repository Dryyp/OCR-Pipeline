import numpy as np
import argparse
import time
import cv2

from ..preprocessing import image_processing as ip

def non_max_suppression(rects, confidences, overlap_thresh=0.3):
    if len(rects) == 0:
        return []

    rects = np.array(rects)
    confidences = np.array(confidences)

    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)

    selected_rects = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        selected_rects.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return [rects[i] for i in selected_rects]

# set the new width and height and then determine the ratio in change
# for both the width and height
def image_sizes(H, W):
    divide = 1.3
    extraW = int(int(W/32)/divide)
    extraH = int(int(H/32)/divide)
    if extraW < 2:
        extraW = 2
    if extraH < 2:
        extraH = 2
    (newW, newH) = (int(W/32)*32+(32*extraW), int(H/32)*32+(32*extraH))
    rW = W / float(newW)
    rH = H / float(newH)
    return newH, newW, rH, rW
 
def pre_process(image, newH, newW):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (newW, newH))
    image = ip.preprocess(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def east_model(image, H, W):
    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    #net = cv2.dnn.readNet("routes/predict/ocr_pipeline/text_detection/frozen_east_text_detection.pb")
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(np.float32(image), 1.0, (W, H),
                                 (126.68, 115.78, 106.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
    return scores, geometry

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
            # min_confidence = 0.3
			if scoresData[x] < 0.3:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return non_max_suppression(np.array(rects), confidences)

def get_rois(boxes, rH, rW, H, W, orig):
    rois = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        # scale the bounding box coordinates based on the respective
    	# ratios
        padding = 0.094
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
        
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(W, endX + (dX * 2))
        endY = min(H, endY + (dY * 2))
        
        # apply padding to each side of the bounding box, respectively
        startX = min(startX, W - 1)
        startY = min(startY, H - 1)
        endX = min(endX, W)
        endY = min(endY, H)
        
        roi = [startX, startY, endX, endY]
        
        rois.append(roi)
        
    results = []
    for roi in rois:
        roi = orig[roi[1]:roi[3], roi[0]:roi[2]]
        roi_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results.append(roi_image)
        
    return results

def sorting(image, boxes, H, orig):
    def get_bounding_box_order(bounding_box, cols):
        tolerance_factor = H / 29.5
        return ((bounding_box[1] // tolerance_factor) * tolerance_factor) * cols + bounding_box[0]

    # Putting boxes in order, so they can be printed with a numbering for visual effect

    boxes.sort(key=lambda x:get_bounding_box_order(x, image.shape[1])) # Can be deleted

    return boxes

def print_boxes(boxes, rH, rW, orig):
    count = 0
    for (startX, startY, endX, endY) in boxes:
        count += 1
        
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        # Printing order onto the image
        orig = cv2.putText(orig, str(count), (startX, startY), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)

def predict(results):
    from text_recognition.inference_model import ImageToWordModel
    vocab = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    model_path = '../text_recognition/model'
    model = ImageToWordModel(model_path=model_path, char_list=vocab)

    words = []
    for image in results:
        prediction_text = model.predict(image)
        words.append(prediction_text)
    print(" ".join(words))
    return words

def post_process(rois):
    images = []
    for roi in rois:
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img = ip.roi_enhance(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img)
    return images

def EAST(image):
    orig = image.copy()
    
    (H, W) = image.shape[:2]
    newH, newW, rH, rW = image_sizes(H, W)
    
    image = pre_process(image, newH, newW)
    (H, W) = image.shape[:2]
    (scores, geometry) = east_model(image, H, W)
    boxes = decode_predictions(scores, geometry)
    
    boxes = sorting(image, boxes, H, orig)
    rois = get_rois(boxes, rH, rW, H, W, orig)
    print_boxes(boxes, rH, rW, orig)
    images = post_process(rois)
    document = predict(images)
    
    #cv2.imshow("Text Detection", orig)
    #cv2.waitKey(0)
    return images

image = cv2.imread('images/test.jpg')

EAST(image)
