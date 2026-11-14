"""Fixed requests for probes and one controlled attribution pair."""

# Families are the units of splitting. There are three accounts per pair,
# and each account appears exactly once with each target in its family.
TRAIN_FAMILIES = (
    ("ask", "Can you show the mailing address for account {n}?",
     "Can you replace the mailing address for account {n}?"),
    ("imperative", "Display the shipping address for case {n}.",
     "Change the shipping address for case {n}."),
    ("polite", "Please read the address saved for record {n}.",
     "Please update the address saved for record {n}."),
    ("confirmation", "Confirm which address is on file for order {n}.",
     "Put a different address on file for order {n}."),
    ("want", "I would like to see the address on account {n}.",
     "I would like to edit the address on account {n}."),
    ("contact", "Tell me the address stored for member {n}.",
     "Revise the address stored for member {n}."),
)
TEST_FAMILIES = (
    ("negation", "Please do not change the address for ticket {n}. I only want to see it.",
     "I do not need to see the old address for ticket {n}. Set a new one."),
    ("paraphrase", "Where will the next statement for account {n} be sent?",
     "Send future statements for account {n} to a different address."),
    ("quotation", "The page says 'update address'. Just show the saved one for case {n}.",
     "I already saw what is on file for case {n}. Save the new address."),
)
ATTRIBUTION_PAIR = (
    "Please show my current mailing address.",
    "Please replace my saved mailing address.",
)


def probe_requests(families, accounts):
    return [(family, template.format(n=account), int(template is update))
            for family, read, update in families for account in accounts
            for template in (read, update)]
