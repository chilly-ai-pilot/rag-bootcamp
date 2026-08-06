import json
with open('results_fixed_100_50_vector.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
for r in data['results']:
    if r['id'] == 8:
        print(f"Query {r['id']}: {r['query']}")
        print(f"\n生成的答案（raw_answer）:")
        print(r['raw_answer'])
        print(f"\n\nCitations:")
        for i, c in enumerate(r['citations'], 1):
            in_answer = '✅' if c['span'] in r['raw_answer'] else '❌'
            print(f"{i}. {in_answer} [{c['source']}]")
            print(f"   Span: '{c['span']}'")
            print()
