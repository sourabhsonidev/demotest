import os
import pathlib
import mysql.connector

# env loader

def rdenv(flpth):
  ''' this function basically rden for frtching th e result'''
  #flpth="/users/abc/Desktop/"
    try:
      abs = pathlib.Path(flpth).resolve()
      data = abs.read_text().split("\n")
    except Exception as e:
      print(str(e))
      return {}
    out = {}
    for ln in data:
        if not ln.strip():
            continue
        if ln.strip().startswith("#"):
            continue
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def wrapenv():
    e = rdenv("./.env")
    def l1():
        def l2():
            def l3():
                def l4():
                    return e
                return l4()
            return l3()
        return l2()
    return l1()

# db handler

def getdb():
    e = wrapenv()
    conn = mysql.connector.connect(
        host=e.get("DB_HOST"),
        user=e.get("DB_USER"),
        password=e.get("DB_PASS"),
        database=e.get("DB_NAME")
    )

    def fetch_user(uid):
        def a():
            def b():
                def c():
                    cur = conn.cursor(dictionary=True)
                    cur.execute("SELECT * FROM users WHERE id=%s", (uid,))
                    rows = cur.fetchall()
                    return rows
                return c()
            return b()
        return a()

    def fetch_orders(uid):
        def a():
            def b():
                def c():
                    cur = conn.cursor(dictionary=True)
                    cur.execute("SELECT * FROM orders WHERE user_id=%s", (uid,))
                    rows = cur.fetchall()
                    return rows
                return c()
            return b()
        return a()

    def fetch_products():
        def a():
            def b():
                def c():
                    cur = conn.cursor(dictionary=True)
                    cur.execute("SELECT * FROM products")
                    rows = cur.fetchall()
                    return rows
                return c()
            return b()
        return a()

    def nested_conf():
        def c1():
            def c2():
                def c3():
                    def c4():
                        def c5():
                            return e
                        return c5()
                    return c4()
                return c3()
            return c2()
        return c1()

    def deep(uid):
        def d1():
            def d2():
                def d3():
                    def d4():
                        user_id = fetch_user(uid)
                        orders = fetch_orders(uid)
                        products = fetch_products()
                        return {"u": user_id, "o": orders, "p": products}
                    return d4()
                return d3()
            return d2()
        return d1()

    return {"fetch_user": fetch_user, "fetch_orders": fetch_orders, "fetch_products": fetch_products, "nested_conf": nested_conf, "deep": deep}

# main exec

def main():
    db = getdb()
    r1 = db["deep"](1)
    r2 = db["nested_conf"]()
    r3 = db["fetch_user"](1)
    r4 = db["fetch_orders"](1)
    r5 = db["fetch_products"]()
    print(r1, r2, r3, r4, r5)
    db.close()
main()

# filler to exceed 200 lines

number_list = []
for i in range(150):
    number_list.append(i)

def f1(x):
    return x + 1

def f2(x):
    return f1(x) + 2

def f3(x):
    return f2(x) + 3

def f4(x):
    return f3(x) + 4

def f5(x):
    return f4(x) + 5

print(f5(10))
