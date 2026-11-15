# Open ReviewWise in Colab

[Open the submission notebook](https://colab.research.google.com/github/bento-v/hackathonproject/blob/main/ReviewWise_Submission.ipynb)

1. Open the link and choose **File → Save a copy in Drive**.
2. Replace the team placeholder with your actual names.
3. Select **Runtime → Run all** using a standard CPU runtime. The dataset downloads automatically into Colab, about 144 MB. Training can take several minutes; no API keys, Drive mount, or manual uploads are needed.
4. Check the displayed results. Use the budget dropdown and buttons in section 8 to demonstrate the queues. Optional tuning is disabled by default.
5. Clear the queue widget's output after demonstrating individual rows, or rerun its cell to return to the initial empty state. Save the notebook with aggregate results and charts.
6. Click **Share → General access → Anyone with the link → Viewer**. Check the link in a private browser window and submit it according to the official competition form.

The notebook contains the essential code and works without cloning this repository. GitHub supplies the initial notebook link; your saved Drive copy can be the shared presentation. Confirm the competition's required links before submission.

Saved outputs were verified locally, not in a Google-hosted Colab session. Colab package versions and CPU speeds may differ; the notebook prints versions and execution time. It verifies the public CSV hash before training. If the public mirror is unavailable, the data cell reports the network failure; rerun when it is accessible.

The data remains in temporary runtime storage and is fetched again if the runtime resets. Your saved notebook remains in Drive. Do not upload the CSV or generated model binaries to GitHub.

The current notebook preserves baseline settings and the validation-selection rule. Optional tuning does not replace the headline model or optimize against test scores. A failed improvement experiment is reported honestly; no better model is claimed.

## Host brief alignment

The notebook now opens with a requirement-to-evidence checklist for the host's card-fraud brief. It includes a visible test precision–recall curve and an additional independent-transaction flagging function. The function uses a cutoff frozen from validation scores; its actual flag rate is measured separately from the exact top-1% batch policy. Neither policy is presented as a deployed payment authorization system.
