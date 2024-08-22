import matplotlib.pyplot as plt
import numpy as np

img = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,1],
    [0,1,0,0,1,0,0,1],
    [0,0,1,0,1,0,0,1],
    [0,1,0,0,1,0,0,0],
    [0,1,0,0,1,0,0,0],
    [0,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,1,0]
])

# img = np.array([
#     [0,0,0,0,0,0,0,0,0],
#     [0,0,0,1,0,0,0,0,0],
#     [0,0,0,1,1,0,0,0,0]
# ])

def get_line(img):
    def get_head(im):
        (m,n) = im.shape
        if n<m: 
            im = im.T
            m,n = n,m
        for d in range(n+m-1):
            for i in range(m):
                if d<i: break
                if d>=n and i<m-(n-d+2): continue
                j = d-i
                # print(f"{d}(dis) = {i}(x) + {j}(y) | need jump: {n-d+2}")
                if im[i,j] == 1: return (i,j)
        raise ValueError("im should not be all zero")
    
    def get_next(im,p_center,p_out=None):
        (i,j) = p_center
        l = [(i-1, j-1), (i-1, j), (i-1, j+1),  (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]
        if p_out is not None:
            l = l + l
            index = l.index(p_out)
            l = l[index+1:index+9]
        for p in l:
            if im[p] == 1:
                return p

    line = [get_head(img)]
    line.append(get_next(img, line[-1], None)) # type: ignore
    while True:
        p = get_next(img, line[-1], line[-2])
        if p == line[0]:
            break
        else:
            line.append(p) # type: ignore
    
    line_img = np.zeros_like(img)
    for p in line:
        line_img[p] = 1
    
    return line_img

plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('Image')
plt.subplot(1,2,2)
plt.imshow(get_line(img),cmap='gray')
plt.title("Edge")

plt.show()