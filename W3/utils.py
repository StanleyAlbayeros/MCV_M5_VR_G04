import cv2

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    overlay = img.copy()
    output = img.copy()
    alpha = 0.9
    cv2.rectangle(overlay, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(overlay, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness, )
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def write_text_two(img50, img101):
    txt_font = cv2.FONT_HERSHEY_PLAIN
    fontsize = 5
    fontcolor = (255,0,255)
    bg_color = (200,200,200)
    fontthickness = 3
    txtSize = cv2.getTextSize("0000", txt_font, fontsize, fontthickness)[0]
    margin = 10
    textX = img101.shape[1] - (txtSize[0])- margin
    textY = margin
    coords = (textX,textY)
    # print(coords)
    

    img50 = draw_text(img50, "R50", txt_font, coords, fontsize, fontthickness, fontcolor, bg_color)
    img101 = draw_text(img101, "R101", txt_font, coords, fontsize, fontthickness, fontcolor, bg_color)
    return img50, img101
