class ReactionsManager:
    def __init__(self, txt_file, sep='>'):
        self.sep = sep
        with open(txt_file, 'r') as f:
            reactions = [line.strip() for line in f.readlines()]
            self.reactions = list(filter(lambda x: x != '', reactions))
        reagents = [[reagent.strip() for reagent in reaction.split(sep)[0].split(',')] for reaction in self.reactions]
        products = [[product.strip() for product in reaction.split(sep)[-1].split(',')] for reaction in self.reactions]
        reagents, products = list(filter(lambda x: len(x) >= 1, reagents)), list(filter(lambda x: len(x) >= 1, products))
        self.reagents = {i: reagent for i, reagent in enumerate(reagents)}
        self.products = {i: product for i, product in enumerate(products)}
        self.title = txt_file.split('/')[-1].split('.')[0]

    @property
    def final_product(self):
        return self.reactions[-1].split(self.sep)[-1]
    
    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, item):
        return self.reactions[item]

    def __repr__(self):
        return f'ReactionsManager({self.title})'

    def get_reagents(self, index):
        return self.reagents[index]

    def get_products(self, index):
        return self.products[index]

    def get_reaction(self, index):
        return self.reactions[index]

    def get_title(self):
        return self.title