from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 定义命名空间
namespace = ("user", "userid_001")

# 存储数据
store.put(namespace, "name", {"values": "张三"})
store.put(namespace, "age", {"values": "18"})
store.put(namespace, "hobby", {"like": ["reading", "gaming"], "dislike": ["eat", "sleep"]})

print(store.get(namespace, "name"))
print(store.search(namespace))

# 删除数据
store.delete(namespace, "name")
print(store.get(namespace, "name"))

# 更新爱好
store.put(namespace, "hobby", {"like": ["阅读", "游戏"]})
print(store.search(namespace))
