using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;

[RequireComponent(typeof(TMP_InputField))]
public class QuestKeyboardTrigger : MonoBehaviour
{
    private TMP_InputField inputField;
    private static TouchScreenKeyboard activeKeyboard;
    private static TMP_InputField activeInputField;

    void Awake()
    {
        inputField = GetComponent<TMP_InputField>();
        inputField.onSelect.AddListener(OnSelectField);
        inputField.onDeselect.AddListener(OnDeselectField);
    }

    private void OnSelectField(string currentText)
    {
        // Close any previously opened keyboard instance to prevent cross-bleeding
        if (activeKeyboard != null && activeKeyboard.active)
        {
            activeKeyboard.active = false;
        }

        activeInputField = inputField;
        activeKeyboard = TouchScreenKeyboard.Open(currentText, TouchScreenKeyboardType.Default);
    }

    private void OnDeselectField(string currentText)
    {
        if (activeInputField == inputField)
        {
            if (activeKeyboard != null)
            {
                activeKeyboard.active = false;
                activeKeyboard = null;
            }
            activeInputField = null;
        }
    }

    void Update()
    {
        // Only update text if THIS specific input field currently owns the active keyboard
        if (activeInputField == inputField && activeKeyboard != null && activeKeyboard.active)
        {
            inputField.text = activeKeyboard.text;
        }
    }
}