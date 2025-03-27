import urllib.request

def GetImage(url):
    url_path = "https://factfury-file.s3.amazonaws.com/uploads/1742668996004-1742668995981-adhaar.jpg"
    img_path = "img.jpg"

    try:
        urllib.request.urlretrieve(url_path, img_path)
        print("Image downlaoded successfully...")
        return img_path


    except Exception as e:
        print(e)
        return None