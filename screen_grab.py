import win32gui
import win32ui
import win32con
import numpy, cv2
from PIL import Image

def convert(bitmap) -> cv2.Mat:
    bmpinfo = bitmap.GetInfo()
    bmpbits = bitmap.GetBitmapBits(True)
    pil_im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpbits, 'raw', 'BGRX', 0, 1)
    pil_array = numpy.array(pil_im)    
    return cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)


def take_screen_shot():
    # https://stackoverflow.com/questions/3586046/fastest-way-to-take-a-screenshot-with-python-on-windows
    w = 1920 # set this
    h = 1080 # set this
    
    hwnd = win32gui.FindWindow(None, "Grand Theft Auto V")
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, (0,0), win32con.SRCCOPY)
    
    image = convert(dataBitMap)

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    return image