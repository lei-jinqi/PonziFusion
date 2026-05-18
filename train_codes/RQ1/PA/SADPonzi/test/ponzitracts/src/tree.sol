pragma solidity ^0.4.24;

contract Ponzi {
    struct User {
        address inviter;
        address itself;
    }
    
    mapping (address => User) Tree;
    address top; 

    function Ponzi() {
        Tree[msg.sender] = User({itself: msg.sender, inviter: msg.sender});
        top = msg.sender;
    }
 
    function enter(address inviter) public {
        uint amount = msg.value;
        Tree[msg.sender] = User({itself: msg.sender, inviter: inviter});

        address next = inviter;
        uint rest = amount;
        while ( next != top){
            uint toSend = rest/2;
            next.send(toSend);
            rest -= toSend;
            next = Tree[next].inviter;
        }
        next.send(rest);
    }
}
 