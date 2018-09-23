import numpy as np
import mnist

imageDimensions = [28,28]
mbSize = 10
num1features  = 3
num2features = 5
#use same size mask for both conv layers
maskL = 5
numHid = 20
numOut = 10
numMP2 = 4*4*num2features
BIAS = True
WSD = 5e-3
EPOCHS = 100
LEARNING_RATE = 5e-3
useFirst = 200
wPen = 5e-5
PRINT = False

trainingImages = mnist.train_images()[10000:10000+useFirst]
trainingLabels = mnist.train_labels()[10000:10000+useFirst]
datasetSize = len(trainingLabels)

def sConv(iL, m):

	iLS = np.shape(iL)
	batchSize = iLS[0]
	iX = iLS[1]
	iY = iLS[2]

	mL = np.shape(m)[0]

	result = []

	for i in range(batchSize):
		iR = np.zeros([iX-mL+1,iY-mL+1])
		inp = iL[i]
		for x in range(iX-mL+1):
			for y in range(iY-mL+1):
				iLpiece = inp[x:x+mL,y:y+mL]
				r = np.sum(iLpiece*m)
				iR[x][y] = r
		result.append(iR)

	return result



#Assumes square image
def conv(inp, weights,bias):

	weightShape = np.shape(weights)

	inputLayers = weightShape[0]
	outputLayers = weightShape[1]

	cMaskL = weightShape[2]

	inpShape = np.shape(inp)

	batchSize = inpShape[0]

	inpL = inpShape[2]

	result = np.zeros([batchSize,outputLayers,inpL-cMaskL+1,inpL-cMaskL+1])

	for i in range(inputLayers):
		rContrib = np.zeros([batchSize,outputLayers,inpL-cMaskL+1,inpL-cMaskL+1])
		for j in range(outputLayers):
			mask = weights[i][j]
			
			inpLayer = inp[:,i,:,:]

			rContrib[:,j] = sConv(inpLayer,mask)

		result = np.add(rContrib,result)

	if(bias):
		result = np.add(result,np.ones(np.shape(result)))

	result = result*(result>0)

	return result


def sMaxPool(layer):

	layerShape = np.shape(layer)
	lX = layerShape[0]
	lY = layerShape[1]

	result = np.zeros([lX/2,lY/2])

	for i in range(lX/2):
		for j in range(lY/2):
			lSub = layer[i*2:(i+1)*2,j*2:(j+1)*2]
			result[i][j] = np.max(lSub)

	return result

#2x2 max pool
def maxPool(inp):

	inpShape = np.shape(inp)

	batchSize = inpShape[0]
	layers = inpShape[1]
	iX = inpShape[2]
	iY = inpShape[3]

	result = np.zeros([batchSize,layers,iX/2,iY/2])

	for i in range(batchSize):
		for j in range(layers):
			l = inp[i][j]
			result[i][j] = sMaxPool(l)

	return result

def sigmoid(x):
	return 1/(1+np.exp(-x))


Wc1 = np.random.normal(scale=WSD, size=[1,num1features,maskL,maskL])
Wc2 = np.random.normal(scale=WSD, size=[num1features,num2features,maskL,maskL])
Wfh = np.random.normal(scale=WSD, size=[numMP2,numHid])
Who = np.random.normal(scale=WSD, size=[numHid,numOut])

def forwardPass(batch,correctPreds):
	c1 = conv(batch,Wc1,BIAS)
	mp1 = maxPool(c1)
	
	c2 = conv(mp1,Wc2,BIAS)
	mp2 = maxPool(c2)

	numFeat = np.size(mp2)/mbSize
	mp2Flat = np.reshape(mp2,[mbSize,numFeat])

	h = np.matmul(mp2Flat,Wfh)
	h = sigmoid(h)

	oI = np.matmul(h,Who)
	oI = np.subtract(oI,np.mean(oI))

	unNormalizedProbs = np.exp(oI)
	sums = np.sum(unNormalizedProbs,1)

	normalizedProbs = []

	for i in range(len(unNormalizedProbs)):
		normalizedProbs.append(unNormalizedProbs[i]/sums[i])

	cost = 0
	correct = 0
	for i in range(len(normalizedProbs)):
		correctProb = np.matmul(np.transpose(normalizedProbs[i]),correctPreds[i])
		cost -= np.log(correctProb)
		if(PRINT):
			print("cProb")
			print(correctProb)
			print("pProb")
			print(np.max(normalizedProbs[i]))
			print("")
		if(correctProb==np.max(normalizedProbs[i])):
			correct+=1

	#       0  1   2  3   4       5 6               7
	return [c1,mp1,c2,mp2,mp2Flat,h,normalizedProbs,cost,correct]

def bpmpToc(c,mp,mpG):
	cShape = np.shape(c)
	bathces = cShape[0]
	layers = cShape[1]
	nX = cShape[2]
	nY = cShape[3]

	cG = np.zeros(np.shape(c))

	for b in range(bathces):
		for l in range(layers):
			for i in range(nX):
				for j in range(nY):
					mx = mp[b][l][i/2][j/2]
					if(mx>0 and c[b][l][i][j]==mx):
						cG[b][l][i][j]=mpG[b][l][i/2][j/2]

	return cG

def backprop(data,c1,mp1,c2,mp2,mp2Flat,h,normalizedProbs,correctPreds):


	oG = np.subtract(normalizedProbs,correctPreds)

	gWho = np.matmul(np.transpose(h),oG)

	hOG = np.matmul(oG,np.transpose(Who))
	sG = (1-h)*h
	hG = hOG*sG

	gWfh = np.matmul(np.transpose(mp2Flat),hG)

	mp2FlatG = np.matmul(hG,np.transpose(Wfh))
	mp2G = np.reshape(mp2FlatG,np.shape(mp2))

	#wathcoutforrelu
	c2G = bpmpToc(c2,mp2,mp2G)

	Wc2G = np.zeros(np.shape(Wc2))
	mp1G = np.zeros(np.shape(mp1))

	Wc2Shape = np.shape(Wc2)
	inLayers = Wc2Shape[0]
	outLayers = Wc2Shape[1]
	maskL = Wc2Shape[2]

	c2Shape = np.shape(c2)
	oX = c2Shape[2]
	oY = c2Shape[3]

	for i in range(inLayers):
		for j in range(outLayers):
			iLs = mp1[:,i]
			oLs = c2G[:,j]
			oLFlat = np.reshape(oLs,np.size(oLs))
			for x in range(maskL):
				for y in range(maskL):
					iSub = iLs[:,x:x+oX,y:y+oX]
					iSubFlat = np.reshape(iSub,np.size(iSub))
					Wc2G[i][j][x][y]=np.dot(iSubFlat,oLFlat)

					weight = Wc2[i][j][x][y]
					contrib = weight*oLs
					full = np.zeros(np.shape(mp1))
					full[:,i,x:x+oX,y:y+oY] = contrib
					mp1G = np.add(mp1G,full)

	c1G = bpmpToc(c1,mp1,mp1G)

	Wc1G = np.zeros(np.shape(Wc1))

	Wc1Shape = np.shape(Wc1)
	inLayers = Wc1Shape[0]
	outLayers = Wc1Shape[1]
	maskL = Wc1Shape[2]

	c1Shape = np.shape(c1)
	oX = c1Shape[2]
	oY = c1Shape[3]

	for i in range(inLayers):
		for j in range(outLayers):
			iLs = data[:,i]
			oLs = c1G[:,j]
			oLFlat = np.reshape(oLs,np.size(oLs))
			for x in range(maskL):
				for y in range(maskL):
					iSub = iLs[:,x:x+oX,y:y+oX]
					iSubFlat = np.reshape(iSub,np.size(iSub))
					Wc1G[i][j][x][y]=np.dot(iSubFlat,oLFlat)

	return [Wc1G, Wc2G,gWfh,gWho]

def encode(cP):
	ret = np.zeros([len(cP),10])
	for i in range(len(cP)):
		ret[i][cP[i]]=1
	return ret


def updateWeights(Wc1L,Wc2L,WfhL,WhoL,imgs,cP):

	c1,mp1,c2,mp2,mp2Flat,h,normalizedProbs,cost,correct = forwardPass(images,cP)
	[Wc1Gradient, Wc2Gradient, WfhGradient, WhoGradient] = backprop(images,c1,mp1,c2,mp2,mp2Flat,h,normalizedProbs,cP)

	Wc1L = np.subtract(Wc1L,LEARNING_RATE*Wc1Gradient)
	Wc1L = (1-wPen)*Wc1L
	Wc2L = np.subtract(Wc2L,LEARNING_RATE*Wc2Gradient)
	Wc2L = (1-wPen)*Wc2L
	WfhL = np.subtract(WfhL,LEARNING_RATE*WfhGradient)
	WfhL = (1-wPen)*WfhL
	WhoL = np.subtract(WhoL,LEARNING_RATE*WhoGradient)
	WhoL = (1-wPen)*WhoL

	return [Wc1L,Wc2L,WfhL,WhoL,cost,correct]

correctPredicitions = [0,1,2,3,4,5,6,7,8,9]

for i in range(EPOCHS):
	tc = []
	acc = []
	for j in range(datasetSize/mbSize):
		images = trainingImages[j*mbSize:(j+1)*mbSize]
		iS = np.shape(images)
		images = np.reshape(images,[iS[0],1,iS[1],iS[2]])
		labels = trainingLabels[j*mbSize:(j+1)*mbSize]
		Wc1,Wc2,Wfh,Who,cost,correct = updateWeights(Wc1,Wc2,Wfh,Who,images,encode(labels))
		tc.append(cost/10)
		acc.append(correct/10.0)
	print("meanWc1")
	print(np.mean(np.abs(Wc1)))
	print("")
	print("Epoch")
	print(i)
	print(np.mean(tc))
	print(np.mean(acc))






