import numpy as np
from metric.classify import Analyzer
from metric.utils import seg_sta


class SGAnalyzer(Analyzer):
    def __init__(self, y_true: np.ndarray, y_score: np.ndarray, epoch: int = 0,
                 threshold: float = 0.5, **kwargs):
        super().__init__(y_true, y_score, epoch=epoch, threshold=threshold, **kwargs)
        self.accumulates = 0

    def analyze(self, threshold: float = 0.5, average: str = 'macro', beta: float = 1.0) -> dict:
        result = super().analyze(threshold, average, beta)
        y_true, y_score = self.raw_data['y_true'], self.raw_data['y_score']

        sg = y_score > 0.5
        gt = y_true > 0.5
        intersection = np.logical_and(sg, gt).sum()
        union = np.logical_or(sg, gt).sum()
        js = intersection / union
        dice = 2 * intersection / (sg.sum() + gt.sum())

        result.update({'JS': js, 'DC': dice, 'score': js + dice})

        return result

    def output_data(self, without_key=False, clean=False):
        out = super().output_data(without_key)
        result = self.result
        sg_out = []

        if clean:
            out.append('{:.2f}%'.format(100 * result['Accuracy']))
            out.append('{:.4f}'.format(result['AUC']))

        for k in seg_sta:
            if k == 'score':
                if without_key:
                    sg_out.append('{:.4f}'.format(result['score']))
                else:
                    sg_out.append('{}:{:.4f}'.format('score', result['score']))
            else:
                if without_key:
                    sg_out.append('{:.2f}'.format(100 * result[k]))
                else:
                    sg_out.append('{}:{:.2f}%'.format(k, 100 * result[k]))
        out = out + sg_out

        return out

    def save_data(self, save_path):
        pass
