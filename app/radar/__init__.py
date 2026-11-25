"""RADAR: view-specific echo captions and a video-text contrastive training skeleton.

The ontology, record, caption and coverage modules have no torch dependency so that the Labelbox tools
(tools/labelbox/) can import them in a light environment. The model, data, loss and train modules need torch.
"""
