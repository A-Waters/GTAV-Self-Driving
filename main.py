import win32gui
import win32ui
import win32con
import PIL, numpy, cv2
from PIL import Image

def take_screen_shot():
    # https://stackoverflow.com/questions/3586046/fastest-way-to-take-a-screenshot-with-python-on-windows
    w = 1920 # set this
    h = 1080 # set this
    # bmpfilenamename = "out.bmp" #set this

    hwnd = win32gui.FindWindow(None, "co-op games.txt - Notepad")
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, (0,0), win32con.SRCCOPY)
    # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
    try:
        bmpinfo = dataBitMap.GetInfo()
        bmparray = numpy.array(dataBitMap.GetBitmapBits()).astype(numpy.uint8)
        pil_im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmparray, 'raw', 'BGRX', 0, 1)
        pil_array = numpy.array(pil_im)
        cv_im = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(e)
        pass
    
    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    print("moved")
    return cv_im

   



image = take_screen_shot()
cv2.imshow("",image)
cv2.waitKey(0)