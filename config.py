class Config:
    # Make this a singleton if I have extra time.
    def __init__(self):
        self.k = 0.5
        # neighborhood definition. 3 for low level and 5 for current level
        self.nl1 = 3
        self.nl2 = 5
        self.nfine = int((self.nl2 ** 2) / 2.)
        self.pad_l1 = int(self.nl1 / 2)
        self.pad_l2 = int(self.nl2 / 2)
        self.padding_l1 = [(self.pad_l1, self.pad_l1), (self.pad_l1, self.pad_l1), (0, 0)]
        self.padding_l2 = [(self.pad_l2, self.pad_l2), (self.pad_l2, self.pad_l2), (0, 0)]
        self.use_yiq = False
