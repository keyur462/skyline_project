# center a given set of polygons to the origin
def initialPolygonCentering(inputWkts, end, start=0):
    centered_wkts=[]
    for i in range(start, end):
        c=inputWkts[i].centroid
        centered_wkts.append(affinity.translate(inputWkts[i], c.centroid.x*-1, c.centroid.y*-1))
    print("{} polygons centered to origin".format(len(centered_wkts)))
    return centered_wkts

# claculate the MBR which can contain all of the polygons in the set.
# It can be boundaries of multiple polygons or a MBR of a single polygon
def findSetMBR(inputWKTs, end, start=0):
    # mbr: minX, minY, maxX, maxY
    mbr=list(inputWKTs[start].bounds)
    for i in range(start+1, end):
        poly_mbr=list(inputWKTs[i].bounds)
        if(poly_mbr[0]<mbr[0]):
            mbr[0]=poly_mbr[0]
        if(poly_mbr[1]<mbr[1]):
            mbr[1]=poly_mbr[1]
        if(poly_mbr[2]>mbr[2]):
            mbr[2]=poly_mbr[2]
        if(poly_mbr[3]>mbr[3]):
            mbr[3]=poly_mbr[3]
    print("Global MBR of [{}-{}] = {}".format(start, end, mbr))
    return mbr

# convert 2D raster into 1D vector string
def convert2DVectorToStr(input):
    out=""
    for i in range(len(input)):
        row_size=len(input)
        for j in range(len(input[i])):
            if input[i][j] != 0:
                if i==len(input)-1 and j==len(input[i])-1:
                    out+=str(i*row_size+j)
                else:
                    out+=str(i*row_size+j)+" "
    return out

# generate list of vectors
def convertWKTListToVectorStrList(g, inputWKTs, end, start=0, cellSelect=1, arg=1):
    vectorList = []
    
    for fid in range(start, end):
        poly = inputWKTs[fid]
        
        if cellSelect == 1:
            g.generateVectorBySubCells(poly, 1, arg)
        else:
            g.generateVectorByCellIntersetingArea(poly, 1, arg)
        vectorList.append(convert2DVectorToStr(g.raster))
        g.resetRaster()
    return vectorList
