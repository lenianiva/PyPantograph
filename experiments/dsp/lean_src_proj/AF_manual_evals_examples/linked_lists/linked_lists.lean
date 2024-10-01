-- Define a singly linked list
inductive Singly_Node (α : Type) : Type
| nil : Singly_Node α
| cons : α → Singly_Node α → Singly_Node α

#check Singly_Node
#check Singly_Node.nil
#check Singly_Node.cons
