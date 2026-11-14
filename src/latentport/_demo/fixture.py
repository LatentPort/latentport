"""Runnable checkout fixture and constrained, model-authored repair validation."""
from decimal import Decimal, ROUND_HALF_UP
import ast
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

FILES = {
"shop/__init__.py": "",
"shop/money.py": '''"""Money boundaries. Domain prices are Decimal MAJOR units; gateways accept integers."""
from decimal import Decimal, ROUND_HALF_UP
EXPONENTS = {"USD": 2, "EUR": 2, "GBP": 2, "JPY": 0, "KWD": 3}

def to_minor_units(amount, currency):
    if currency not in EXPONENTS:
        raise ValueError("Unsupported currency")
    value = Decimal(str(amount))
    if not value.is_finite() or value < 0:
        raise ValueError("Invalid monetary amount")
    return int((value * (Decimal(10) ** EXPONENTS[currency])).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def from_minor_units(amount, currency):
    if not isinstance(amount, int):
        raise TypeError("Gateway amounts must be integers")
    return Decimal(amount) / (Decimal(10) ** EXPONENTS[currency])
''',
"shop/checkout.py": '''"""Checkout orchestrates pricing and constructs a gateway payment request."""
from decimal import Decimal
from shop.money import to_minor_units
from shop.pricing import subtotal

def checkout(unit_price, quantity, currency="USD"):
    total = subtotal(unit_price, quantity)
    amount_minor = int(total)
    return {"amount_minor": amount_minor, "currency": currency, "description": "Atlas checkout"}
''',
"shop/pricing.py": '''"""All pricing functions return Decimal values in major currency units."""
from decimal import Decimal

def subtotal(unit_price, quantity):
    price = Decimal(str(unit_price))
    if not price.is_finite() or price < 0:
        raise ValueError("Price must be finite and nonnegative")
    if type(quantity) is not int or not 1 <= quantity <= 100:
        raise ValueError("Quantity must be an integer between 1 and 100")
    return price * quantity

def discount(total, percent):
    rate = Decimal(str(percent))
    if not 0 <= rate <= 100:
        raise ValueError("Invalid percentage")
    return total * (Decimal(100) - rate) / Decimal(100)

def shipping(total, free_threshold, fee):
    return Decimal(0) if total >= free_threshold else Decimal(str(fee))
''',
"shop/inventory.py": '''"""Reservation bookkeeping: quantities are item counts, never money."""
from dataclasses import dataclass

@dataclass
class Stock:
    on_hand: int
    reserved: int = 0
    @property
    def available(self):
        return self.on_hand - self.reserved
    def reserve(self, count):
        if count <= 0 or count > self.available:
            raise ValueError("Insufficient stock")
        self.reserved += count
    def release(self, count):
        if count <= 0 or count > self.reserved:
            raise ValueError("Invalid release")
        self.reserved -= count
    def fulfill(self, count):
        self.release(count)
        self.on_hand -= count
''',
"shop/gateway.py": '''"""Local gateway simulator. All input amounts use integer minor units."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Receipt:
    payment_id: str
    amount_minor: int
    currency: str
    status: str

class Gateway:
    def __init__(self):
        self.payments = {}
    def charge(self, amount_minor, currency, idempotency_key):
        if type(amount_minor) is not int or amount_minor < 0:
            raise ValueError("Invalid amount")
        if not idempotency_key:
            raise ValueError("Missing idempotency key")
        payload = (amount_minor, currency)
        if idempotency_key in self.payments:
            old_payload, receipt = self.payments[idempotency_key]
            if old_payload != payload:
                raise ValueError("Idempotency conflict")
            return receipt
        receipt = Receipt("pay_" + str(len(self.payments) + 1), amount_minor, currency, "captured")
        self.payments[idempotency_key] = (payload, receipt)
        return receipt
''',
"shop/orders.py": '''"""Order transitions are independent of payment amount conversion."""
from dataclasses import dataclass

TRANSITIONS = {
    "created": {"reserved", "cancelled"},
    "reserved": {"paid", "cancelled"},
    "paid": {"shipped", "refunded"},
    "shipped": {"returned"},
    "returned": {"refunded"},
    "refunded": set(),
    "cancelled": set(),
}

@dataclass
class Order:
    order_id: str
    state: str = "created"
    version: int = 0
    def transition(self, destination, expected_version):
        if expected_version != self.version:
            raise ValueError("Stale order version")
        if destination not in TRANSITIONS[self.state]:
            raise ValueError("Invalid order transition")
        self.state = destination
        self.version += 1
''',
"tests/test_incident.py": '''import unittest
from shop.checkout import checkout

class Incident(unittest.TestCase):
    def test_two_headphones(self):
        self.assertEqual(checkout("59.95", 2, "USD")["amount_minor"], 11990)
    def test_three_notebooks(self):
        self.assertEqual(checkout("12.50", 3, "USD")["amount_minor"], 3750)

if __name__ == "__main__": unittest.main()
''',
"README.md": """# Atlas checkout service

Run: python -m unittest discover -s tests -v

The incident: two headphones at $59.95 each should charge $119.90.
The checkout screen displays the correct cart total, but the gateway receives 119
minor units and charges $1.19. This is a local demonstration application with no
network connections or real payment credentials.

The repair must retain the public checkout(unit_price, quantity, currency) API.
Do not change pricing logic, the gateway contract, or test expectations to hide the bug.
The pricing domain uses Decimal major units. Gateway requests use integer minor units.
Reuse the currency conversion helper in shop/money.py at the correct boundary.
""",
}

# These executable tests supply meaningful repository context, including edge cases
# in adjacent systems. They are normal tests, not repeated prompt filler.
def build_fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    files = dict(FILES)
    for currency, exponent in {"USD":2,"EUR":2,"GBP":2,"JPY":0,"KWD":3}.items():
        rows = ['import unittest', 'from decimal import Decimal',
                'from shop.money import to_minor_units, from_minor_units', '',
                f'class Money{currency}(unittest.TestCase):']
        for n, price in enumerate(['0','0.01','0.49','1','1.25','9.99','12.50','59.95','119.90','999.99','1000','9999.99']):
            minor = int((Decimal(price)*(10**exponent)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
            rows += [f'    def test_amount_{n}(self):',
                     f'        self.assertEqual(to_minor_units(Decimal("{price}"), "{currency}"), {minor})',
                     f'        self.assertEqual(from_minor_units({minor}, "{currency}"), Decimal("{minor}") / Decimal("{10**exponent}"))','']
        files[f'tests/test_money_{currency.lower()}.py']='\n'.join(rows)
    rows=['import unittest','from decimal import Decimal','from shop.pricing import subtotal, discount, shipping','','class Pricing(unittest.TestCase):']
    for n,(price,quantity) in enumerate(( (p,q) for p in ['0.01','1.25','12.50','59.95','99.99'] for q in [1,2,3,10])):
        rows += [f'    def test_cart_{n}(self):',f'        total = subtotal("{price}", {quantity})',
                 f'        self.assertEqual(total, Decimal("{Decimal(price)*quantity}"))',
                 '        self.assertEqual(discount(total, 10), total * Decimal("0.90"))',
                 '        self.assertEqual(shipping(total, Decimal("50"), "4.99"), Decimal(0) if total >= 50 else Decimal("4.99"))','']
    files['tests/test_pricing.py']='\n'.join(rows)
    rows=['import unittest','from shop.orders import Order, TRANSITIONS','','class Orders(unittest.TestCase):']
    for n,(source,dest) in enumerate(( (a,b) for a in ['created','reserved','paid','shipped','returned','refunded','cancelled'] for b in ['created','reserved','paid','shipped','returned','refunded','cancelled'])):
        rows += [f'    def test_transition_{n}(self):',f'        order = Order("order-{n}", state="{source}")']
        if dest in {'created':{'reserved','cancelled'},'reserved':{'paid','cancelled'},'paid':{'shipped','refunded'},'shipped':{'returned'},'returned':{'refunded'},'refunded':set(),'cancelled':set()}[source]:
            rows += [f'        order.transition("{dest}", 0)',f'        self.assertEqual(order.state, "{dest}")','        self.assertEqual(order.version, 1)']
        else:
            rows += ['        with self.assertRaises(ValueError):',f'            order.transition("{dest}", 0)']
        rows += ['']
    files['tests/test_orders.py']='\n'.join(rows)
    rows=['import unittest','from shop.inventory import Stock','','class Inventory(unittest.TestCase):']
    for n,count in enumerate([1,2,3,5,8,10,20,50,99,100]):
        rows += [f'    def test_reservation_{n}(self):',f'        stock = Stock({count*2})',f'        stock.reserve({count})',
                 f'        self.assertEqual(stock.available, {count})',f'        stock.fulfill({count})',
                 f'        self.assertEqual(stock.on_hand, {count})','        self.assertEqual(stock.reserved, 0)','']
    files['tests/test_inventory.py']='\n'.join(rows)
    files['tests/test_gateway.py']='''import unittest
from shop.gateway import Gateway

class GatewayTests(unittest.TestCase):
    def test_retry_returns_same_receipt(self):
        gateway = Gateway()
        first = gateway.charge(11990, "USD", "order-1")
        self.assertIs(first, gateway.charge(11990, "USD", "order-1"))
    def test_conflicting_retry_is_rejected(self):
        gateway = Gateway()
        gateway.charge(11990, "USD", "order-1")
        with self.assertRaises(ValueError):
            gateway.charge(119, "USD", "order-1")
    def test_decimal_gateway_amount_rejected(self):
        with self.assertRaises(ValueError):
            Gateway().charge(119.90, "USD", "order-1")
'''
    for name,text in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding='utf-8')
    # Put the integration contract and incident last for a readable source bundle.
    names=sorted(files,key=lambda name:(name in ('README.md','shop/money.py','shop/checkout.py','tests/test_incident.py'),name))
    bundle='\n\n'.join(f'FILE: {name}\n```\n{files[name]}\n```' for name in names)
    return bundle,files


def run_tests(root, pattern='test_*.py'):
    proc=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',pattern,'-v'],
                        cwd=root,capture_output=True,text=True,timeout=30,encoding='utf-8',errors='replace')
    text=proc.stdout+proc.stderr
    match=re.search(r'Ran (\d+) tests?',text)
    return {'passed':proc.returncode==0,'returncode':proc.returncode,'test_count':int(match.group(1)) if match else None,'output':text}


def extract_json(text):
    text=text.rsplit('</think>',1)[-1]
    decoder=json.JSONDecoder()
    for i,c in enumerate(text):
        if c=='{':
            try:
                obj,_=decoder.raw_decode(text[i:])
                if isinstance(obj,dict): return obj
            except ValueError: pass
    raise ValueError('Model did not return a JSON object')


def validate_patch(payload):
    if payload.get('file')!='shop/checkout.py': raise ValueError('Patch must target shop/checkout.py')
    old=payload.get('old','').strip();new=payload.get('new','').strip()
    if old!='amount_minor = int(total)': raise ValueError('Patch did not identify the incident line')
    # Only the existing, trusted conversion helper can be called. Never execute
    # arbitrary model-generated Python in the presenter process.
    actual=ast.dump(ast.parse(new),include_attributes=False)
    allowed=[ast.dump(ast.parse(code),include_attributes=False) for code in
             ['amount_minor = to_minor_units(total, currency)','amount_minor = to_minor_units(total, currency=currency)',
              'amount_minor = to_minor_units(amount=total, currency=currency)']]
    if actual not in allowed: raise ValueError('Patch is outside the permitted single-line money conversion repair')
    return old,new


def validate_cases(payload):
    cases=payload.get('cases')
    if not isinstance(cases,list) or not 2<=len(cases)<=8: raise ValueError('Expected 2..8 test cases')
    for case in cases:
        if set(case)!={'price','quantity','currency','expected_minor'}: raise ValueError('Unexpected test case fields')
        p=Decimal(str(case['price']));q=case['quantity'];c=case['currency'];e=case['expected_minor']
        if not p.is_finite() or not 0<p<=10000 or type(q)!=int or not 1<=q<=100 or c not in {'USD','JPY','KWD'} or type(e)!=int:
            raise ValueError('Invalid generated test inputs')
        expected=int((p*q*(10**{'USD':2,'JPY':0,'KWD':3}[c])).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
        if e!=expected: raise ValueError('Generated expectation disagrees with the documented money contract')
    if not {'USD','JPY'} <= {c['currency'] for c in cases}: raise ValueError('Tests must cover USD and JPY')
    return cases


def test_source(cases):
    return ('import unittest\nfrom shop.checkout import checkout\n\nclass GeneratedRegression(unittest.TestCase):\n'+
            ''.join(f'    def test_case_{i}(self):\n        self.assertEqual(checkout({str(c["price"])!r}, {c["quantity"]}, {c["currency"]!r})["amount_minor"], {c["expected_minor"]})\n'
                    for i,c in enumerate(cases)))


def verify(root, output, patch, cases):
    old,new=validate_patch(patch);cases=validate_cases(cases)
    work=output/'repaired-checkout';shutil.copytree(root,work)
    generated=test_source(cases)
    (work/'tests/test_generated.py').write_text(generated,encoding='utf-8')
    before=run_tests(work,'test_generated.py')
    if before['passed']: raise ValueError('Generated tests did not reproduce the bug')
    file=work/'shop/checkout.py';text=file.read_text()
    if text.count(old)!=1: raise ValueError('Patch match is not unique')
    file.write_text(text.replace(old,new),encoding='utf-8')
    after=run_tests(work)
    if not after['passed']: raise ValueError('Patched checkout failed the test suite: '+after['output'])
    return {'before':before,'after':after,'patch':{'file':'shop/checkout.py','old':old,'new':new},
            'generated_tests':generated,'cases':cases,'checkout_path':str(work)}
