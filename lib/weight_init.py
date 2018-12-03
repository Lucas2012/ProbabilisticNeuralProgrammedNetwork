def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        except:
            pass
    elif classname.find('Linear') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)

        except:
            pass
    elif classname.find('Embedding') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        except:
            pass