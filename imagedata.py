# Imagedata : manipulating image data
import numpy as np
import matplotlib.pyplot as plt


def get_cropi(data,center=None,width=None,height=None,point1=None,point2=None,point=None):
	# Prepare Output
	x = np.arange(0,data.shape[1])
	y = np.arange(0,data.shape[0])
	[XX,YY] = np.meshgrid(x,y)
	cropi = (slice(None,None),slice(None,None))

	# Option 1 -- center width and height
	if center is not None:
		if width is None and height is None: return cropi
		if width is not None and height is None: height = width
		if width is None and height is not None: width = height
		xmin = max(0, center[0] - int(width/2.0))
		xmax = min(x[-1], xmin + width)
		ymin = max(0, center[1] - int(height/2.0))
		ymax = min(y[-1], ymin + height)

	# Option 2 -- point1 and point 2
	elif point1 is not None and point2 is not None:
		xmin = max(min(point1[0], point2[0]) , 0    )
		xmax = min(max(point1[0], point2[0]) , x[-1]) + 1
		ymin = max(min(point1[1], point2[1]) , 0    )
		ymax = min(max(point1[1], point2[1]) , y[-1]) + 1

	# Option 3 -- point and width and height
	elif point is not None:
		if width is None and height is None: return cropi
		if width is not None and height is None: height = width
		if width is None and height is not None: width = height
		xmin = max(point[0]         , 0)
		ymin = max(point[1]         , 0)
		xmax = min(point[0] + width , x[-1])
		ymax = min(point[1] + height, y[-1])
	else:
		return cropi

	# Return a np array of true false
	return (slice(ymin,ymax),slice(xmin,xmax))



def main():
	# Tests of get_cropi
	data = np.ones((512,256))
	cropi = get_cropi(data,center=(50,100),width=10,height=50)
	print('center width height:',data[cropi].shape)
	cropi = get_cropi(data)
	print('nothing',data[cropi].shape)
	cropi = get_cropi(data,point1=(50,50),point2=(100,100))
	print('two points',data[cropi].shape)
	cropi = get_cropi(data,point=(50,50),width=50)
	print('point and width',data[cropi].shape)
	plt.imshow(data[cropi],cmap='gray')
	plt.show()



if __name__ == '__main__':
	main()