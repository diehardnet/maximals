import torch


class HardenedIdentity(torch.nn.Identity):
    __MAX_LAYER_VALUE = 1024.0
    __MIN_LAYER_VALUE = -__MAX_LAYER_VALUE
    __TO_NAN_VALUE = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Keep the layer original behavior
        input = super(HardenedIdentity, self).forward(input=input)
        # Smart hardening
        return torch.clamp(
            input=torch.nan_to_num(input, nan=self.__TO_NAN_VALUE, posinf=self.__MAX_LAYER_VALUE,
                                   neginf=self.__MIN_LAYER_VALUE),
            min=self.__MIN_LAYER_VALUE, max=self.__MAX_LAYER_VALUE
        )
