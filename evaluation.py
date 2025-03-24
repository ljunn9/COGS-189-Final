accuracy = accuracy_score(all_true, all_preds)
print(f" Accuracy: {accuracy:.2f}\n")

print("Classification Report:")
print(classification_report(all_true, all_preds, target_names=["Left", "Select", "Right"]))

cm = confusion_matrix(all_true, all_preds)
labels_display = ["Left", "Select", "Right"]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels_display, yticklabels=labels_display)
plt.title("Confusion Matrix (K-Fold) for Simon's Data")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
