from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=50):

	#initialize the ID for the next unique object, as well
	#as 2 ordered dicts to keep track of mapping object IDs
	#to their centroid and number of consecutive frames they
	#have beenn marked as disappeared, respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
	#store the number of max consecutive frames an object
	#is allowed to be marked as disappeared for until we 
	#need to deregister that object's ID from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):

	#we use the next available object ID to store the centroid
	#of the next object we're registering
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	#when deregistering an object we delete the object ID from
	#both our objects and disappeared frames dicts	
	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
	#check to see if the list of input bounding box rectangles
	#is empty
		if len(rects) == 0:
		#loop over existing tracked objects and mark them as
		#disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
			#if we have reached a max number of consecutive
			#frames an object has been missing, dereg it
			#bye felicia
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects

		#init an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		#loop over the bouding box rects
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
		#use the bounding box coordinates to then derive the centroid
			cX = int((startX) + endX / 2.0)
			cY = int((startY) + endY / 2.0)
			inputCentroids[i] = (cX, cY)

		#if we are not currently tracking any objects take the input
		#centroids and register each of them

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		#else, as we're currently tracking objs we need to try to
		#match the input centroids to the existing object centroids

		else:
		#grab the set of object ID/centroid pairs
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			#compute distance between each pair of obj centroids and
			#input centroids, respectively
			#-----------------------------
			#our goal will be to match an input centroid to an existing
			#object centroid

			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			#in order to match these we must find the smallest value in each row
				#then sort the row indexes based on their minimum values,
				#so that the row with the smallest value is
				#at the front of the index list

			rows = D.min(axis=1).argsort()

			#now, we similarly find the smallest value in each column
			#and then sorting using the previously computed row index list

			cols = D.argmin(axis=1)[rows]

			#in order to decide if we're to update/register/deregister
			#an object we need to track which of the row&column indexes
			#we've already examined

			usedRows = set()
			usedCols = set()

			#loop over the combination of the row&column index tuples
			for (row, col) in zip(rows, cols):
				#ignore already examined rows and columns
				if row in usedRows or col in usedCols:
					continue
				#else, grab the objID for the current row, set its
				#new centroid, and reset the disappeared counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				#indicate that we have examined each of the row&column
				#indexes, respectively
				usedRows.add(row)
				usedCols.add(col)
			#compute both the row and column index we have NOT examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(1, D.shape[1])).difference(usedCols)

			#if we end up with the same or more object centroids as 
			#input centroids, we need to check if some of these objects
			#have potentially disappeared
			if D.shape[0] >= D.shape[1]:
				#loop over unused row indexes
				for row in unusedRows:
					#grab the objectID for the resp. row index and
					#increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					#check to see if the number of consecutive
					#frames the object has been marked "disappeared"
					#warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			#else, if the number of input centroids is greater than the number
			#of existing object centroids we need to register each new input
			#centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		#return the set of trackable objects
		return self.objects
