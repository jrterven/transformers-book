"""A deliberately small intent task with disjoint wording and entity groups.

Every account and address is fictional. The held-out split is for reporting
the fixed run; its outcomes never select a checkpoint or hyperparameter.
"""

LABELS = ('LOOKUP', 'UPDATE')
TRAIN = (
    ('Please show the saved address for {account}.',
     'Find the current mailing address for {account}.',
     'What address is on file for {account}?',
     'I need to read the address registered to {account}.',
     'Look up the delivery address for {account}.',
     'Tell me which address is stored for {account}.'),
    ('Please replace the address for {account} with 17 Cedar Road.',
     'Change the mailing address of {account} to 28 Willow Lane.',
     '{account} moved. Save 39 Oak Street as the new address.',
     'Update the registered address for {account} to 41 Pine Avenue.',
     'The new address for {account} is 52 Maple Road; store it.',
     'Write 63 Birch Lane over the old address for {account}.'),
)
HELD = (
    ('Could you retrieve the address associated with {account}?',
     'Where should a parcel for {account} be delivered according to the record?',
     'Read back the stored location for {account}.',
     'Which mailing location do you have for {account}?'),
    ('{account} no longer lives at the old location. Record 74 Ash Lane instead.',
     'Set the delivery location for {account} to 85 Elm Street.',
     'Correct the address entry for {account}: it should be 96 Beech Road.',
     'Overwrite the mailing location of {account} with 107 Fir Avenue.'),
)


def examples(split):
    if split not in ('train', 'held'):
        raise ValueError('split must be train or held')
    templates = TRAIN if split == 'train' else HELD
    accounts = range(8) if split == 'train' else range(100, 104)
    rows = []
    for label, forms in enumerate(templates):
        for family, template in enumerate(forms):
            for account in accounts:
                request = template.format(account=f'account {account:03d}')
                rows.append({'split': split, 'label': label, 'family': f'{split}-{label}-{family}',
                             'account': account, 'request': request,
                             'prompt': 'Classify the request as LOOKUP or UPDATE. Return the label only.\n'
                                       f'Request: {request}\nIntent:',
                             'response': ' ' + LABELS[label],
                             'rejected': ' The intent is ' + LABELS[label] + '.'})
    return rows
