import torch
from edge0.cpu import BailingConfig
from edge0.model import BailingCPUModel

def test_model_has_24_layers_and_schedule():
    c=BailingConfig()
    m=BailingCPUModel(c,{})
    assert len(m.layers)==24
    assert [x.is_mla for x in m.layers[:8]]==[False,False,False,True,False,False,False,True]

def test_router_and_expert_store_are_present_after_moe_cutover():
    c=BailingConfig()
    m=BailingCPUModel(c,{})
    assert m.layers[0].experts is None
    assert m.layers[1].experts is not None
    assert m.layers[1].router is not None
