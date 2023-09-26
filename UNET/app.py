import torch
import argparse
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from unet_model import UNET
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
app = Flask(__name__)

@app.route('/polyp_mask/', methods=['GET'])
def infer_mask():
    try:
        data = request.get_json()
        base64_image = data.get('image')
        image_data = base64.b64decode(base64_image)
        img = Image.open(BytesIO(image_data)).resize((256, 256))
        img = transforms.ToTensor()(img).unsqueeze(0).to(device)

        mask = model(img).squeeze(0, 1)
        mask = mask.cpu().detach().numpy()
        base64_image = base64.b64encode(mask).decode()

        return jsonify({"result": base64_image}), 200
    except Exception as e:
        return jsonify({'Error':str(e)}), 500



if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--weight', type=str, default='./weight/UNET_model.pt')
    parser.add_argument('-hs', '--host', dest='host', type=str, required=True)
    parser.add_argument('-ps', '--port', dest='port', type=int, required=True)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNET(3, 1)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.to(device)
    app.run(host=args.host, port=args.port)