from pycocoevalcap.eval import COCOEvalCap
from coco import COCO

coco = COCO('annotations.json')
cocoRes = coco.loadRes('generated.json')

cocoEval = COCOEvalCap(coco, cocoRes)
# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

print(len(cocoEval.params['image_id']))
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()
