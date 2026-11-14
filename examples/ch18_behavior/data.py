"""Authored requests for a simulated address desk. No real accounts or actions."""

GUIDE = (
    "Office routing guide\n"
    "A: Read the address without changing it.\n"
    "B: Replace the address with a new one.\n"
    "C: Ask which action the customer wants if it is unclear.\n\n"
    "Request: Show the current address.\nAnswer: A\n"
    "Request: Put a different address on file.\nAnswer: B\n"
    "Request: I have a question about my address.\nAnswer: C\n\n"
)

# Score every paired wording, including pairs on which the model makes a mistake.
PAIRS = (
    ("Please show my mailing address.", "Please replace my mailing address."),
    ("Please display my delivery address.", "Please change my delivery address."),
    ("Could you read my billing address?", "Could you update my billing address?"),
)

# Define the action by whether a read or change is explicitly requested.
# Deliberately include misleading words and ambiguity on the third slice.
SLICES = {
    "Direct": (
        ("Please show my mailing address.", "A"),
        ("Please replace my mailing address.", "B"),
        ("Can you display the address you have for me?", "A"),
        ("Please update my delivery address to 10 Elm Lane.", "B"),
        ("I need help with my mailing address.", "C"),
        ("Could we talk about my saved address?", "C"),
    ),
    "Paraphrased": (
        ("What address do you currently have on file for my orders?", "A"),
        ("Which address will you send the next invoice to?", "A"),
        ("Send future orders to 4 Birch Court instead.", "B"),
        ("Use 50 Oak Avenue for my invoices from now on.", "B"),
        ("My address might need some attention.", "C"),
        ("Is there something I need to do about my address?", "C"),
    ),
    "Distracting context": (
        ("I am not asking you to update anything. Just tell me the address on file.", "A"),
        ("The form says 'change address', but I only want to see my current one.", "A"),
        ("I already looked up the old address. Save 14 Pine Road as the new one.", "B"),
        ("Don't look it up again. Replace it with 21 Cedar Street.", "B"),
        ("The buttons say 'show' and 'change'. I need help with the address.", "C"),
        ("My friend told me to update the address. Can we discuss my options first?", "C"),
    ),
}


def prompt(request):
    return GUIDE + f"Request: {request}\nAnswer:"
