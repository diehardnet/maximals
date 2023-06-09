import torch


class HardenedIdentity(torch.nn.Identity):
    # TODO: we need to profile many inputs to define this maximum value
    __MAX_LAYER_VALUE = 1024.0
    __MIN_LAYER_VALUE = -__MAX_LAYER_VALUE
    __TO_NAN_VALUE = 0.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Keep the layer original behavior and implement smart hardening
        return torch.clamp(  # Remove too large values, positive and negative
            input=torch.nan_to_num(  # Remove all nan values
                super(HardenedIdentity, self).forward(input=input),
                nan=self.__TO_NAN_VALUE, posinf=self.__MAX_LAYER_VALUE,
                neginf=self.__MIN_LAYER_VALUE
            ),
            min=self.__MIN_LAYER_VALUE, max=self.__MAX_LAYER_VALUE  # Other values that are too large and are not inf
        )
