from src.amap.test_amap import test_amap_extraction
import seaborn as sns
import matplotlib.pyplot as plt

# test amap extraction
diff_res = {}
for n in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    test_result = test_amap_extraction('cuda', n_samples=n, fp16=False)
    diff_res[n]=test_result['diff_input_output']
    print("Diff: ", test_result['diff_input_output'])


print(diff_res)

sns.lineplot(x=list(diff_res.keys()), y=list(diff_res.values()))

# Customize the plot (add titles, labels, etc.) if needed
plt.title("Tracking input/output diff regarding to the number of samples")
plt.xlabel("Number of samples")
plt.ylabel("Input/output diff")

# Save the plot to a file (replace 'output_plot.png' with your desired filename)
plt.savefig('diff_test.png')
