class KerasHook:
    def __init__(self, keras):
        self.keras = keras
        if not hasattr(keras, "_hooked"):
            self._hook_layers()
            self._hook_sequential()
            self.keras._hooked = True

    def _hook_layers(self):
        for layer_cls in filter_layers(self.keras.layers, tfe.keras.layers):
            layer_cls = add_constructor_registration(layer_cls)

    def _hook_sequential(self):
        seq_cls = getattr(self.keras, "Sequential")
        setattr(seq_cls, "share", share)
        setattr(seq_cls, "serve", serve)
        setattr(seq_cls, "stop", stop)
