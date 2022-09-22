from skimage.measure import euler_number

def eul(im,conn=3):

    # im: numpy array 
    conn = euler_number(im,connectivity=conn)
    return conn
