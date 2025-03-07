# SPDX-License-Identifier: Apache-2.0
"""
This file contains a new class `KVLookupBufferBase` that allows developers to 
think of KV cache operations as inserting new KV cache entries (`insert`) 
into the lookup buffer and querying existing KV caches (`drop_select`) 
from the lookup buffer.

All distributed communications are abstracted behind this class.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class KVLookupBufferBase(ABC):
    """
    Abstract base class for a lookup buffer.

    This class provides an abstraction for a key-value (KV) cache lookup buffer.
    
    The key of the lookup buffer:
    - input_tokens: token IDs of the request
    - roi: a binary mask on top of input_tokens.
      - Purpose of roi: Since KV cache may only be available for a subset of 
        tokens in the input (for example, when vLLM is connected to an external 
        KV cache service), roi specifies the subset of tokens that the KV cache 
        is associated with.
      - NOTE: roi can be further extended to describe which part of KV the 
        current process is holding (each process may only hold a part of KV 
        due to TP and PP). This is not implemented for now.
        
    The value of the lookup buffer:
    - key: the key tensor in the KV cache
    - value: the value tensor in the KV cache
    - hidden: the final hidden state generated by model forwarding. This allows 
      vLLM to bypass further model forwarding by transmitting the hidden state.
    """

    @abstractmethod
    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        """Insert into the lookup buffer.
        
        The functionality is similar to the following python statement
        ```
        buffer[input_tokens, roi] = [key, value, hidden]
        ```
        
        FIXME: in the future, we should only have two arguments, key and value,
        where key is a tensor dict and value is a tensor dict.
        
        FIXME: we should transmit both sampler outputs and the hidden states.

        Args:
            input_tokens (torch.Tensor): token IDs.
            roi (torch.Tensor): A binary mask on top of the input tokens
            key (torch.Tensor): The key tensor in the KV cache.
            value (torch.Tensor): The value tensor in the KV cache.
            hidden (torch.Tensor): The final hidden state tensor generated 
                                   during model forwarding to bypass model 
                                   forwarding.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        """Select and *drop* KV cache entries from the lookup buffer.
        
        The functionality is similar to the following python statements
        ```
        ret = buffer.pop(input_tokens, roi)
        return ret
        ```
        
        If `input_tokens` and `roi` is `None`, it means selecting any of the
        KV caches in the buffer, return, and remove it from the buffer, useful
        when offloading KV cache to KV cache storage service.

        Args:
            input_tokens (torch.Tensor): token IDs.
            roi (torch.Tensor): A binary mask on top of the input tokens

        Returns:
            List[Optional[torch.Tensor]]: A list of tensors. Can be None.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the buffer and release resources.

        This method is responsible for cleaning up resources related to the 
        lookup buffer when it is no longer needed.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError
