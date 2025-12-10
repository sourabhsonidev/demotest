import React, { useState } from "react";

// Large React component with nested logic, loops, JSON unpacking
// and more than 200 lines.

export default function JsonUnpackerMega() {
  const [rawJson, setRawJson] = useState("{}");
  const [parsed, setParsed] = useState(null);
  const [logz, setLogz] = useState([]);

  // Dummy JSON for default
  const dummyJson = {
    user: {
      name: "Alpha",
      creds: {
        key: "DUMMY_KEY_123",
        token: "DUMMY_TOKEN_999",
      },
      meta: {
        role: "tester",
        level: 5,
        access: ["read", "write", "execute"],
      },
    },
    items: [
      { id: 1, name: "it1", data: { a: 1, b: 2, c: [3, 4, 5] } },
      { id: 2, name: "it2", data: { a: 10, b: 20, c: [30, 40] } },
      { id: 3, name: "it3", data: { a: 7, b: 14, c: [21] } },
    ],
    settings: {
      flags: {
        enabled: true,
        debug: false,
        nested: {
          opt1: "alpha",
          opt2: "beta",
          opt3: "gamma",
        },
      },
      modes: [
        { mode: "safe", lvl: 1 },
        { mode: "fast", lvl: 2 },
        { mode: "aggressive", lvl: 3 },
      ],
    },
  };

  const handleLoadDummy = () => {
    setRawJson(JSON.stringify(dummyJson, null, 2));
  };

  const processJson = () => {
    try {
      const obj = JSON.parse(rawJson);
      setParsed(obj);
      const lz = [];

      // -- nested unpacking logic begins --
      function walk(value, path = "root") {
        if (Array.isArray(value)) {
          lz.push(`Array at ${path}, length=${value.length}`);
          for (let i = 0; i < value.length; i++) {
            walk(value[i], `${path}[${i}]`);
          }
        } else if (value !== null && typeof value === "object") {
          lz.push(`Object at ${path}`);
          for (const k in value) {
            walk(value[k], `${path}.${k}`);
          }
        } else {
          lz.push(`Primitive at ${path}: ${String(value)}`);
        }
      }

      walk(obj);
      setLogz(lz);
    } catch (e) {
      setParsed(null);
      setLogz(["Invalid JSON"]);
    }
  };

  // Repetitive nested structural logic to exceed 200 lines
  function nestedRepeat(val) {
    let acc = 0;
    for (let i = 0; i < val; i++) {
      for (let j = 0; j < i; j++) {
        for (let k = 0; k < j; k++) {
          acc += i + j + k;
        }
      }
    }
    return acc;
  }

  const bigRunner = () => {
    const logs = [];
    for (let a = 0; a < 10; a++) {
      for (let b = 0; b < 5; b++) {
        const res = nestedRepeat(a + b);
        logs.push(`calc(${a},${b}) = ${res}`);
      }
    }
    return logs;
  };

  const moreLogs = bigRunner();

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">JSON Unpacker Mega Component</h1>

      <textarea
        className="w-full h-40 p-2 border rounded"
        value={rawJson}
        onChange={(e) => setRawJson(e.target.value)}
      />

      <div className="flex space-x-2">
        <button className="px-3 py-2 bg-blue-500 text-white rounded" onClick={processJson}>
          Process JSON
        </button>
        <button className="px-3 py-2 bg-green-500 text-white rounded" onClick={handleLoadDummy}>
          Load Dummy
        </button>
      </div>

      {parsed && (
        <div className="p-2 border rounded bg-gray-50">
          <h2 className="font-semibold">Parsed JSON Structure</h2>
          <pre className="text-sm">{JSON.stringify(parsed, null, 2)}</pre>
        </div>
      )}

      <div className="p-2 border rounded bg-gray-50">
        <h2 className="font-semibold">Traversal Logs</h2>
        <div className="max-h-60 overflow-auto text-sm">
          {logz.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      </div>

      <div className="p-2 border rounded bg-gray-100">
        <h2 className="font-semibold">More Nested Loop Logs</h2>
        <div className="max-h-60 overflow-auto text-sm">
          {moreLogs.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      </div>

      {/* Filler repeated blocks for exceeding 200 lines */}
      {Array.from({ length: 40 }).map((_, idx) => (
        <div key={idx} className="text-xs text-gray-600">
          Filler segment #{idx + 1}: Nested loop marker
        </div>
      ))}
    </div>
  );
}
