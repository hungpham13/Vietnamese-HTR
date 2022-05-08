
class Evaluator(object):
    def __init__(self):
        self.det_labels = []
        self.true_labels = []

    def update(self, targets, outputs):
        """extend a set of targets and outputs to evaluation data"""
        for output in outputs:
            self.det_boxes.append(output['boxes'])
            self.det_labels.append(output['labels'])
            self.det_scores.append(output['scores'])
        for target in targets:
            self.true_boxes.append(target['boxes'])
            self.true_labels.append(target['labels'])

    def summarize(self):
